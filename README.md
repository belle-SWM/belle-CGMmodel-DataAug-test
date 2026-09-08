# 2024/11/04 blood glucose model (v006)
修改程式如下:
1. 更新模型最佳化使用的衡量指標為specificity
2. 改變模型最佳化更新條件
3. 修改predict 函式，至少3個以上的10秒ECG片段為好的品質，才給出分類結果
4. 修改CNN模型layer設計, relu變為leaky relu, 並修改模型參數初始化

呼叫方法請參考v005版說明

# 2024/08/01 blood glucose model (v005)
## 使用方法如下請參考:
* Step 1. 基本參數設定    
```
  uuid='1732'  ##設定uuid 
  basepath='C:\\Users\\User\\Desktop\\GlucoseModelProject\\Model'  ###血糖數值對應之ECG資料，分析得到特徵檔案會存放的位置
  srj_db_path="C:\\Users\\User\\Desktop\\DataDB\\1732"   ###srj檔放置路徑
  glucosedata_path='C:\\Users\\User\\Desktop\\GlucoseModelProject\\GlucoseDataCSV' ###APP收到的血糖數值，整理成CSV檔後存放路徑
 ```   

* Step 2. 血糖模型建立
```
  status, errorcode, message = BuildModel(uuid,basepath,srj_db_path,glucosedata_path)  ###建立個人化模型(自動根據是否有低血糖資料，決定訓練中高血糖模型或是高中低血糖模型)
  print('status:',str(status),' error code:',errorcode,' message:',message)
```
  
* Step 3. 輸入ECG進行血糖類別分類          
```
    GlucosePredictor_Obj=GlucosePredictor(uuid,basepath) 
    errorcode, message, glucose_category=GlucosePredictor_Obj.predict(ecgdata)  ##輸入大於一分鐘的ECG訊號陣列，輸出血糖類別，-1: bad signal, 0:normal, 1:high, 2:low
    print('glucose_category:',glucose_category)
    print('errorcode:',str(errorcode),' message:',message)  
```
* 其他會用到的函式
  ```
    errorcode,message=Clean_Data(uuid,basepath) ##刪除uuid所有過去資料
    print('errorcode:',errorcode)
    print('message:',message)
    
    errorcode,message=Clean_Model(uuid,basepath) ##刪除uuid所有過去訓練之模型
    print('errorcode:',errorcode)
    print('message:',message)
  ```
    
* status狀態說明:
  ```
  status=-1: 模型訓練失敗
  status=0: 模型訓練成功但績效沒有比之前版本好
  status=1: 模型訓練成功且績效比之前版本好
  ``` 
* 錯誤碼(errorcode)定義:
  ```
  "0": No problem, the model has been built.
  "-100": An error occurs in the data-parsing function: fail to read glucose csv files.
  "-101": An error occurs in the data-parsing function: ecg data is too short or no motion data!
  "-102": An error occurs in the data-parsing function: no glucose csv files exist!
  "-200": An error occurs in the feature_extraction function: the number of R wave peak is zero!
  "-201": An error occurs in the feature_extraction function: fail to seperate data into high, low, normal categories!
  "-300": An error occurs in the data_arrangement function: fail to move data to normal, high and low categories!
  "-400": An error occurs in the BuildModel_ThreeClasses function: No training data.
  "-401": An error occurs in the BuildModel_ThreeClasses function: No testing data.
  "-402": An error occurs in the BuildModel function: No enough high glucose data.
  "-500": An error occurs in the BuildModel_TwoClasses function: No training data.
  "-501": An error occurs in the BuildModel_TwoClasses function: No testing data.
  "-600": An error occurs in the predict function of the GlucoseThreeClassesPredictor class: Quality of ECG signal is bad in some 10-second segments!
  "-601": An error occurs in the predict function of the GlucoseThreeClassesPredictor class: Data is too short, the lenth of input ecg data is shorter than 1 minute long!
  "-602": "Model has not been built!"
  "-700": An error occurs in the predict function of the GlucoseTwoClassesPredictor class: Quality of ECG signal is bad in some 10-second segments!
  "-701": An error occurs in the predict function of the GlucoseTwoClassesPredictor class: Data is too short, the lenth of input ecg data is shorter than 1 minute long!
  "-800": An error occurs in the Clean_Data function: some data for uuid xxx has not deleted clearly!
  "-801": An error occurs in the Clean_Model function: No built model for uuid xxx exists, it can not be cleaned!
  "-802": An error occurs in the Clean_Data function: Model for uuid xxxx has not deleted clearly!

  ```
  


# 2024/07/03 blood glucose model (v004)
## 使用方法如下請參考:
* Step 1. 基本參數設定    
```
  uuid='1732'  ##設定uuid 
  basepath='C:\\Users\\User\\Desktop\\GlucoseModelProject\\Model'  ###血糖數值對應之ECG資料，分析得到特徵檔案會存放的位置
  srj_db_path="C:\\Users\\User\\Desktop\\DataDB\\1732"   ###srj檔放置路徑
  glucosedata_path='C:\\Users\\User\\Desktop\\GlucoseModelProject\\GlucoseDataCSV' ###APP收到的血糖數值，整理成CSV檔後存放路徑
 ```   

* Step 2. 血糖模型建立
```
  errorcode, message = BuildModel(uuid,srj_db_path,glucosedata_path,basepath)  ###建立個人化模型(自動根據是否有低血糖資料，決定訓練中高血糖模型或是高中低血糖模型)
  print('error code:',errorcode,' message:',message)
```
  
* Step 3. 輸入ECG進行血糖類別分類          
```
  if(int(errorcode)==0):
        if(message=="Model with three classes has been built!"):  ###高中低血糖分類模型
            GlucosePredictor_Obj=GlucoseThreeClassesPredictor(basepath) 
            errorcode, message, glucose_category=GlucosePredictor_Obj.glucose_predict(uuid,ecgdata)  ##輸入大於一分鐘的ECG訊號陣列，輸出血糖類別，-1: bad signal, 0:normal, 1:high, 2:low
            print('glucose_category:',glucose_category)
            print('errorcode:',str(errorcode),' message:',message)
        else:   ###高中血糖分類模型
            print('Perform high prediction!')
            GlucoseTwoClassesPredictor_Obj=GlucoseTwoClassesPredictor(basepath) 
            errorcode, message, glucose_category=GlucoseTwoClassesPredictor_Obj.glucose_predict(uuid,ecgdata)  ##輸入大於一分鐘的ECG訊號陣列，輸出血糖類別，-1: bad signal, 0:normal, 1:high
            print('glucose_category:',glucose_category)
            print('errorcode:',str(errorcode),' message:',message)
```

* 備註: 錯誤碼(errorcode)定義
  ```
  "0": No problem, the model has been built.
  "-100": An error occurs in the data-parsing function: fail to read glucose csv files.
  "-200": An error occurs in the feature_extraction function: the number of R wave peak is zero!
  "-201": An error occurs in the feature_extraction function: fail to seperate data into high, low, normal categories!
  "-300": An error occurs in the data_arrangement function: fail to move data to normal, high and low categories!
  "-400": An error occurs in the BuildModel_ThreeClasses function: No training data.
  "-401": An error occurs in the BuildModel_ThreeClasses function: No testing data.
  "-500": An error occurs in the BuildModel_TwoClasses function: No training data.
  "-501": An error occurs in the BuildModel_TwoClasses function: No testing data.
  "-600": An error occurs in the predict function of the GlucoseThreeClassesPredictor class: Quality of ECG signal is bad in some 10-second segments!
  "-601": An error occurs in the predict function of the GlucoseThreeClassesPredictor class: Data is too short, the lenth of input ecg data is shorter than 1 minute long!
  "-700": An error occurs in the predict function of the GlucoseTwoClassesPredictor class: Quality of ECG signal is bad in some 10-second segments!
  "-701": An error occurs in the predict function of the GlucoseTwoClassesPredictor class: Data is too short, the lenth of input ecg data is shorter than 1 minute long!

  ```
