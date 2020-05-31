import ntpath
import os
import subprocess
from multiprocessing import Pool

import requests
import tempfile
import logging
import yaml
import datetime
from pydub import AudioSegment

# from infer_wrapper import SEInferenceWrapper
from yandex_cloud_service import YaCloudService


# def execute_nn_denoiser(src_file_path, out_file_path):
#     # Init NN service wrapper
#     model_dir = 'models'
#     nn_infer_wrapper = SEInferenceWrapper(model_dir)
#     nn_infer_wrapper.denoise_file(src_file_path, out_file_path)


class DFLDenoisingMessageHandler:
    '''
    Class to create object with response for DFL denoising request
    '''

    __DENOISER_SAMPLE_RATE = 16000
    __MAX_AUDIO_DURATION_SECONDS = 600

    def __init__(self, config_file_path):
        self.__logger = logging.getLogger()

        with open(config_file_path, 'r') as stream:
            try:
                config = yaml.safe_load((stream))
            except yaml.YAMLError as exc:
                self.__logger.error(f"Can't parse config file")
                self.__logger.error(exc)

        # Init NN service wrapper
        # model_dir = 'models'
        # self.__nn_infer_wrapper = SEInferenceWrapper(model_dir)
        self.__temp_files = []

        # Init Ya cloud storage service
        ya_bucket_name = config['ya_cloud_storage']['bucket_name']
        ya_creds_obj = config['ya_cloud_storage']
        self.__cloud_storage = YaCloudService(ya_bucket_name, ya_creds_obj)

    @staticmethod
    def __get_file_name_from_path(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    @staticmethod
    def __upload_and_save_file(file_url, out_file_path):
        r = requests.get(file_url, allow_redirects=True)
        open(out_file_path, 'wb').write(r.content)

    @staticmethod
    def __get_file_name_from_url(url):
        res = url.rsplit('/', 1)[1]
        return res

    def __seconds_to_span(self, secs) -> str:
        res = str(datetime.timedelta(seconds=secs))
        return res

    @staticmethod
    def call_denoiser(in_file_path: str, out_file_path: str):
        process = subprocess.Popen(["python",
                                    # 'senet_infer_cli.py',
                                    'senet_infer_cli.py',
                                    '--in_file', in_file_path,
                                    '--out_file', out_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stderr_str = stderr.decode("utf-8")
        if len(stderr_str) > 0:
            print('------------------------ STDERR ------------------------\n' + stderr_str)
        assert os.path.isfile(out_file_path)

    def get_response_obj(self, req_obj):
        def cleanup_temp_files():
            for file_path in self.__temp_files:
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Load file from url
        src_file_urls_list = req_obj['FileUrlsList']
        src_file_paths_list = []

        sox_target_audio_params = ['-r',  str(self.__DENOISER_SAMPLE_RATE),
                                   '-b', '32',
                                   '-e', 'float']

        for src_file_url in src_file_urls_list:
            long_file_name = self.__get_file_name_from_url(src_file_url)
            question_mark_index = long_file_name.find('?')
            if question_mark_index > -1:
                long_file_name = long_file_name[0:question_mark_index]
            long_file_path = os.path.join(tempfile.gettempdir(), long_file_name)
            long_file_path = os.path.join('/denoising/audio/', long_file_name)
            if not os.path.exists(long_file_path):
                self.__logger.info(f'TRY: Save initial file to {long_file_path}')
                self.__upload_and_save_file(src_file_url, long_file_path)
                self.__logger.info(f'SUCCESS: Initial file saved to {long_file_path}')
                self.__temp_files.append(long_file_path)

            # Convert initial file to wav with target sample rate
            long_file_wav_path = long_file_path.replace('.mp3', '.wav')
            self.__logger.info(f'TRY: Convert initial file to target audio params to file: {long_file_wav_path}')
            process = subprocess.Popen(["sox",
                                        long_file_path,
                                        *sox_target_audio_params,
                                        long_file_wav_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.communicate()
            self.__logger.info(f'SUCCESS: File converted to "{long_file_wav_path}"')
            self.__temp_files.append(long_file_path)
            self.__temp_files.append(long_file_wav_path)
            src_file_paths_list.append(long_file_wav_path)
            self.__logger.info(f'Src wav file path is {long_file_wav_path}')
            assert os.path.isfile(long_file_wav_path)

        denoised_file_urls_list = []
        for src_file_path in src_file_paths_list:
            denoised_full_file_path = src_file_path.replace('.wav', '_denoised.wav')
            self.__logger.info(f'Target denoised full file path is {denoised_full_file_path}')

            src_wav_file_obj = AudioSegment.from_wav(src_file_path)

            # Split initial file to parts
            cur_src_file_part_start = 0
            cur_src_file_part_end = self.__MAX_AUDIO_DURATION_SECONDS
            if cur_src_file_part_end > src_wav_file_obj.duration_seconds:
                cur_src_file_part_end = src_wav_file_obj.duration_seconds

            denoised_file_parts_paths_list = []
            it = 1

            while cur_src_file_part_start < src_wav_file_obj.duration_seconds:
                src_file_part_path = src_file_path.replace('.wav', f'_part_{it}.wav')
                denoised_file_part_path = denoised_full_file_path.replace('.wav', f'_part_{it}.wav')

                self.__logger.debug(f'Denoised part file path is {denoised_file_part_path}')
                self.__logger.debug(f'Src part stamps is {cur_src_file_part_start}s - {cur_src_file_part_end}s')
                self.__logger.debug(f'Src part file path is {src_file_part_path}')

                process = subprocess.Popen(["sox",
                                            src_file_path,
                                            src_file_part_path,
                                            'trim',
                                            self.__seconds_to_span(cur_src_file_part_start),
                                            self.__seconds_to_span(cur_src_file_part_end)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.communicate()

                self.__logger.info(f'Call denoising of file part "{src_file_part_path}" to: "{denoised_file_part_path}"')
                self.call_denoiser(src_file_part_path, denoised_file_part_path)
                denoised_file_parts_paths_list.append(denoised_file_part_path)

                # Move parts splitting window forward
                it += 1
                cur_src_file_part_start = cur_src_file_part_end
                cur_src_file_part_end += self.__MAX_AUDIO_DURATION_SECONDS
                if cur_src_file_part_end > src_wav_file_obj.duration_seconds:
                    cur_src_file_part_end = src_wav_file_obj.duration_seconds

            denoised_mp3_file_path = denoised_full_file_path.replace('.wav', '.mp3')
            self.__logger.info(f'Concat part files to one denoised file')
            process = subprocess.Popen(["sox",
                                        *denoised_file_parts_paths_list,
                                        denoised_mp3_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.communicate()

            self.__logger.debug(f'Resulting file exported to {denoised_mp3_file_path}')
            assert os.path.isfile(denoised_mp3_file_path)

            cloud_save_file_name = denoised_mp3_file_path.replace('\\', '/').split('/')[-1]
            self.__logger.info(f'TRY: Save file to cloud as: "{cloud_save_file_name}"')
            saved_file_url = self.__cloud_storage.save_object_to_storage(denoised_mp3_file_path, cloud_save_file_name)
            self.__logger.info(f'SUCCESS: Saved denoised file URL is: "{saved_file_url}"')
            denoised_file_urls_list.append(saved_file_url)
            self.__temp_files.append(denoised_mp3_file_path)

        res = {'FileUrlsList': denoised_file_urls_list}
        cleanup_temp_files()
        return res
