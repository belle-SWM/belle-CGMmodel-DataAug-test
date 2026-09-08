import os
import zipfile
import json
from datetime import datetime
from natsort import natsorted
from .srj_time_parse import file_process

def mkdir(output_folder):
    folder = os.path.exists(output_folder)  # 檢查路徑是否存在
    if not folder:
        os.makedirs(output_folder)


def unzip(zip_folder, save_folder, download_list=[]):
    if len(download_list) > 0:
        list_zip = []
        for fn in download_list:
            if fn in os.listdir(zip_folder):
                list_zip.append(fn)
    else:
        list_zip = os.listdir(zip_folder)
    for f in list_zip:
        if f.endswith('zip'):
            with zipfile.ZipFile(zip_folder + '/' + f, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.evt') | name.endswith('.srj'):
                        zf.extract(name, path=save_folder)   # 解壓縮指定檔案至save_folder


def get_selected_date_list(all_date_list, start_date, end_date):
    s = all_date_list.index(start_date)
    e = all_date_list.index(end_date)
    return all_date_list[s:e + 1]


def convert_timestamp(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    return dt_object


def srj2ecg_txt(srj_path, txt_path):
    with open(srj_path) as f:
        srj_lines = f.readlines()
        f.close()

    ecgs = []
    for srj_line in srj_lines:
        ecgs.extend(json.loads(srj_line)["rows"]["ecgs"])

    # save ECG in txt
    with open(txt_path, 'w') as f:
        for line in ecgs:
            f.write(f"{line}\n")

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(100,30))
    # plt.plot(ecgs, linewidth=0.7)
    # plt.show()
    return


if __name__ == '__main__':

    """
    ## 兩種下載方法 ##
    方法一: 設定好日期範圍(start_date, end_date),將日期內 "所有檔案" 從SWM.datacenter解壓，將.evt, .srj 儲存至指定資料夾
    方法二: 設定好日期範圍(start_date, end_date),指定檔名清單,將清單內 "指定檔案" 從SWM.datacenter解壓，將.evt, .srj 儲存至指定資料夾
    
    # zip_Folder : SWM_Datacenter ECG 壓縮檔案位置
    # unzip_Folder : ECG 解壓後存檔位置
    # in_folder : ECG 資料預處理輸入 (同unzip_Folder)
    # out_folder : ECG 資料預處理輸出
    # txt_file_path : 將所有10秒的訊號組成完整ECG,儲存成txt檔
    # IRR_Check_FileList : 輸入c++判斷心率的檔案清單
    

    
    
    目標 : 產生"檔名", "路徑" txt檔案 給cpp讀取

    """

    # ----- Setting----------------------------------------------------
    # 是否要下載並解壓縮 ECG 檔案
    do_unzip = True

    # ECG 壓縮檔路徑
    zip_Folder = r'H:\.shortcut-targets-by-id\1Mc_sTYrGzDau1JPki2AQDpV4FeX5tKu3\SWM_DataCenter\Health_Server/'
    unzip_Folder = r'H:\我的雲端硬碟\SWM_data\unzip_ECG/'
    # unzip_Folder = r'C:\Users\u1042\BJM_code\SWM_Algorithm_Test\download_data\unzip_ECG/'

    # 設定日期範圍

    # start_date = '20211201'
    # end_date = '20211231'

    # start_date = '20220101'
    # end_date = '20220228'

    # start_date = '20220701'
    # end_date = '20220930'

    start_date = '20230203'
    end_date = '20230205'

    # 是否有指定檔案清單
    DownLoad_list = os.listdir(r'C:\Users\SWM-Benjamin\Desktop\BJM\UpsideDown\Combined\Twave_reverse/')

    # 是否執行 Beatinfo_Srj_Time_Parse.py 進行資料前處理
    do_preprocessing = True
    # ECG 資料預處理 輸入及輸出路徑
    in_folder = unzip_Folder
    out_folder = r'H:\我的雲端硬碟\SWM_data\ParseData/'
    # out_folder = r'C:\Users\u1042\BJM_code\SWM_Algorithm_Test\download_data\preprocess_ECG/'
    # out_folder = r'C:\Users\u1042\BJM_code\BJM\hb_practice\download_data\data/'

    # 是否輸串接後完整的 ECG.txt 檔案
    do_concat = True

    # 是否寫入IRR_Check_FileList.txt 與C++程式配合聯動
    yield_list_to_cpp = False
    # ------------------------------------------------------------------

    # 生成資料的日期清單
    date_list = os.listdir(zip_Folder)
    date_list = natsorted(date_list, key=lambda y: y.lower())
    date_list = get_selected_date_list(date_list, start_date, end_date)

    # 解壓縮並存檔
    if do_unzip:
        for fold in date_list:
            mkdir(unzip_Folder + fold)
            unzip(zip_folder=zip_Folder + fold,
                  save_folder=unzip_Folder + fold,
                  download_list=DownLoad_list)

    # 執行srj_time_parse.py 並輸出 concated ECG txt 檔
    if do_preprocessing:
        for fold in date_list:
            print(f'--- Date : {fold} -------------------------------------------------------------------------------')
            import_path = in_folder + fold
            export_path = out_folder + fold
            file_process(import_path, export_path)

    # 由.srj 輸串接後完整的 ECG.txt 檔案
    """ example:
    91_1662945579163_546C0EDE3E64.srj  >>  91_20220912091939.txt  (有轉換timestamp:2022.09.12, 09:19:39)
    """
    if do_concat:
        for fold in date_list:
            srj_file_folder = out_folder + fold
            txt_file_path = out_folder + fold
            srj_list = [srj_file for srj_file in os.listdir(srj_file_folder) if srj_file.endswith('srj')]

            for srj in srj_list:
                start = srj.find('_', 0) + 1
                end = srj.find('_', start)
                timestamp = int(srj[start:(end-3)])
                time = convert_timestamp(timestamp)
                time = str(time).replace("-", "")
                time = str(time).replace(":", "")
                time = str(time).replace(" ", "")
                txt_name = srj[:start] + time + '.txt'
                txt_path = txt_file_path + '/' + txt_name

                print(f'Start concatenating ECG : Date - {fold}')
                srj2ecg_txt(srj_file_folder + '/' + srj, txt_path)

    # 是否寫入IRR_Check_FileList.txt 與C++程式配合聯動
    if yield_list_to_cpp:
        for fold in date_list:
            srj_file_folder = out_folder + fold
            txt_file_path = out_folder + fold
            srj_list = [srj_file for srj_file in os.listdir(srj_file_folder) if srj_file.endswith('srj')]

            for srj in srj_list:
                start = srj.find('_', 0) + 1
                end = srj.find('_', start)
                timestamp = int(srj[start:(end - 3)])
                time = convert_timestamp(timestamp)
                time = str(time).replace("-", "")
                time = str(time).replace(":", "")
                time = str(time).replace(" ", "")
                txt_name = srj[:start] + time + '.txt'
                txt_path = txt_file_path + '/' + txt_name

                # 寫入 IRR_Check_FileList.txt
                with open(r'C:\Users\u1042\BJM_code\SWM_Algorithm_Test\IRR_Check_FileList\FileListFor_IRR_Check.txt',
                          'a') as f:
                    f.write('./download_data/preprocess_ECG/' + fold + '/' + txt_name + '\n')

                # 寫入 FileNameList.txt
                with open(r'C:\Users\u1042\BJM_code\SWM_Algorithm_Test\IRR_Check_FileList\FileNameList.txt', 'a') as f:
                    f.write(txt_name[:-4] + '\n')
