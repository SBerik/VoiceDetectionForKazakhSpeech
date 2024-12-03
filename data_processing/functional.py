import os
import soundfile as sf
import numpy as np
import librosa
from energy_vad import read_wav # !!! warning !!!!
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

def get_counts_with_p(files, pattern):
    return sum(1 for f in files if f.endswith(f'.{pattern}'))

def get_files_with_p(files, pattern):
    return [f for f in files if f.endswith(f'.{pattern}')]

def getPlot (time_frame, s, fs, d): 
    min_length = min(len(time_frame), len(d))
    time_frame = time_frame[:min_length]
    d = d[:min_length]

    plt.plot(time_frame, d, color = 'red')
    plt.show()

def merge_audio_files(input_folder, output_folder, group_size):
    '''
    Если аудиозаписей нечетное количество то оставшиеся мы группируем 
    '''
    audio_files = get_files(input_folder)
    # audio_files.sort() 
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, len(audio_files), group_size):
        group_files = audio_files[i:i + group_size]
        combined_data = []

        total_file_name = ''
        for file_name in group_files:
            file_path = os.path.join(input_folder, file_name)
            f_name_without_ex = os.path.splitext(file_name)[0]
            if total_file_name:
                total_file_name += '_' + f_name_without_ex
            else:
                total_file_name += f_name_without_ex
            samplerate, data = read_wav(file_path)
            
            combined_data.append(data)

        combined_data = np.concatenate(combined_data)

        output_file_name = f'{total_file_name}.flac'
        output_file_path = os.path.join(output_folder, output_file_name)
        sf.write(output_file_path, combined_data, samplerate)

# # Пример использования функции
# input_folder = 'C:\\Users\\b.smadiarov\\Diploma\\VD-KazakhSpeech\\rawaudio'  # Путь к папке с аудиофайлами
# output_folder = 'C:\\Users\\b.smadiarov\\Diploma\\VD-KazakhSpeech\\rawaudio'  # Путь для сохранения объединённых файлов
# group_size = 2  # Укажите количество записей в группе

# merge_audio_files(input_folder, output_folder, group_size)

# VALIDATE DATASET
def count_flac(tracklist):
    return sum(1 for file in tracklist if file.endswith('.flac'))

def count_duplicates(flac_files):
    used = set()
    duplicates = 0
    for file in tqdm(flac_files):
        name_without_ext = os.path.splitext(os.path.basename(file.replace('\\', '/')))[0]
        if name_without_ext in used:
            duplicates += 1
        used.add(name_without_ext)
    return duplicates

def count_duplicates_between_folders(flac_files):
    def extractInfo(file):
        name_without_ext = os.path.splitext(os.path.basename(file.replace('\\', '/')))[0]
        label = file.replace('\\', '/').split('/')[3]  # train, dev, test
        typee = file.replace('\\', '/').split('/')[4] # radio, tv ...
        return name_without_ext, label, typee
        
    used = defaultdict(list)
    duplicates = 0  
    for file in tqdm(flac_files):
        name_without_ext, label, typee = extractInfo(file)
        if name_without_ext in used and any(label != f[1] and f[2] == typee for f in used[name_without_ext]):
            duplicates += 1
        used[name_without_ext].append((file, label, typee))
    return duplicates

def getDoubleExt(file_list):
    pattern = re.compile(r'(.+)\.flac\.flac$')  # Ищем именно .flac.flac
    double_ext_files = [file for file in file_list if pattern.match(file)]
    double_ext_files = sorted(double_ext_files)
    return double_ext_files

def extract_filenames(file_paths):
    pattern = re.compile(r'([^\\/]+)(?=\.\w+(\.\w+)*$)')
    filenames = []
    for path in file_paths:
        match = pattern.search(path)
        if match:
            filenames.append(match.group(1).replace('.flac', ''))
    return filenames
    
def count_double_ext_matches(file_list, double_ext_files):
    double_ext_fs_names = set(extract_filenames (double_ext_files))
    count = 0
    res = []
    for file_pt in tqdm(file_list):
        file_name = os.path.splitext(os.path.basename(file_pt.replace('\\', '/')))[0]   

        # os.path.splitext(os.path.basename('F:/ISSAI_KSC2_unpacked/ISSAI_KSC2/Dev/talkshow/415465.flac.flac'.replace('\\', '/')))[0]  
        # '415465.flac'
        # os.path.splitext(os.path.basename('F:/ISSAI_KSC2_unpacked/ISSAI_KSC2/Dev/talkshow/415465.flac'.replace('\\', '/')))[0]    
        # '415465'
        
        if file_name in double_ext_fs_names:
            count += 1
            res.append(file_name)
            
    return count, res

def rename_double_ext_audios(audios_with_double_ext) -> None:
    for file_path in audios_with_double_ext:
        file_path = file_path.replace('\\', '/')
        dir_name, file_name_with_ex = os.path.split(file_path)
        file_name = file_name_with_ex.split('.')[0]
        if file_name_with_ex.endswith('.flac.flac'):
            new_file_name = file_name + '.flac' 
            new_file_path = dir_name + '/' + new_file_name
            os.rename(file_path, new_file_path)

def process_dataframe(dataframe):
    print("Count of intervals: ", dataframe.shape[0]) 
    speech_in_ds = dataframe[dataframe['speech'] == 1].utt_time.sum()
    non_speech_in_ds = dataframe[dataframe['speech'] == 0].utt_time.sum()
    total_speech_time = speech_in_ds + non_speech_in_ds
    print("Total time of speech (s): ", speech_in_ds)
    print("Total time of non-speech (s): ", non_speech_in_ds)
    print("Total time (s) in dataset: ", total_speech_time)
    return speech_in_ds, non_speech_in_ds


def get_total_time_duration(flac_files):
    total_duration = 0
    for file in tqdm(flac_files):
        file = file.replace('\\', '/')
        with sf.SoundFile(file) as audio:
            duration = len(audio) / audio.samplerate  
            total_duration += duration
    return total_duration

def findJsonPare(flac_files, json_files, jsons_path): 
    # json_paths: paths['KS2_annotation_json']
    flac_files = glob (glob_paths['KS2_raw']) 
    flac_files = [t.replace('\\', '/') for t in flac_files]
    json_files = glob (glob_paths['KS2_jsons']) 
    json_files = [x.replace('\\', '/') for x in json_files]
    missing_json_files = [] 
    json_set = set(json_files) 
    for t in tqdm(flac_files): 
        t = t.replace('\\', '/') 
        json_from_flac = jsons_path + re.sub(r'.*ISSAI_KSC2', '', t) 
        json_from_flac = json_from_flac.replace('.flac', '.json')
        if json_from_flac not in json_set: 
            missing_json_files.append(t) 
    return missing_json_files

def mapper_from_json_to_flac (json_files, base_path):
    res = []
    for json in json_files:
        json =  json.replace('\\', '/')
        flac_file = (base_path +  re.sub(r'.*KazakhSpeech/data/annotation/Json/', '', json)).replace('.json', '.flac')
        res.append(flac_file)
    return res

def delete_files(file_list):
    for file_path in file_list:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Ошибка при удалении {file_path}: {e}")
        else:
            print(f"Файл {file_path} не найден.")

def getSamplesWithOneClassesInTheSameFolder (tracklist):        
    def isSingleClassSample(t) -> bool:
        zeroes = pd.read_csv(t.replace('.flac', '.csv'))['speech'].eq(0).any()
        ones = pd.read_csv(t.replace('.flac', '.csv'))['speech'].eq(1).any()
        return (ones and zeroes)
    single_samples = []
    for t in tqdm(tracklist):
        t = t.replace('\\', '/')
        if not isSingleClassSample(t):
            single_samples.append(t)
    return single_samples