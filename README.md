# UAI-CUP-2017

## Requirement

Python 2.7

Keras 2.x

sklearn

pandas

## Run

UAI-CUP-2017目录下的文件为各种不同的ML/DL模型代码，运行分为两种方式：

- 只训练，不产生提交结果（用于测试代码的正确性和了解模型效果）,例如：

		python embedding_mlp.py train

- 训练且产生测试集结果用于提交，例如：

		python embedding_mlp.py submit

	

## 测试结果
### 不使用POI的ID
|模型/方法	|Train MAE|Dev MAE|Pubulic Test MAE|备注说明|
|---------|:---:|:----:|:--:|------|
|MLP       |0.4543|0.4189|2.2674|不使用One-Hot|
|MLP       |0.4530|0.4184|2.2315|对周几和小时使用One-Hot处理|
|ETR       |0.1448|0.5916|**2.1252**|同上|

### 使用POI的ID
|模型/方法	|Train MAE|Dev MAE|Pubulic Test MAE|备注说明|
|---------|:---:|:----:|:--:|------|
|MLP      |0.4235|0.3992|**2.1209**|对离散型特征作One-Hot处理|
|Emb_MLP  |0.4225|0.3966|2.3590|对离散型特征作Embeddding处理|
|ETR      |0.1448|0.5871|**2.1316**|对离散型特征作One-Hot处理|

### 将所有订单状态都认为是需求量
|模型/方法	|Train MAE|Dev MAE|Pubulic Test MAE|备注说明|
|---------|:---:|:----:|:--:|------|
|MLP      |0.6760|0.6795|**2.0317**|对离散型特征作One-Hot处理|
|Emb_MLP  |0.6847|0.6811|2.2911|对离散型特征作Embeddding处理|
|ETR      |0.2232|0.9417|**2.1140**|对离散型特征作One-Hot处理|
|MLP      |0.6874|0.6925|**2.0148**|对结果四舍五入|
|MLP      |0.7056|0.6856|**2.1154**|取下整|
|MLP      |0.6473|0.6827(0.6800)|**2.0266**|两段one-hot|
|MLP      |0.6769|0.6879(0.6839)|**xxxx**|两段one-hot|
|MLP      |1.6831|1.4100(1.3947)|**2.0336**|去掉需求大于9的数据，每个区间随机选9000条|
|MLP      |0.6613|0.6859(0.680244)|**2.020768816**|POI聚类用0,1表示|
|MLP      |1.6831|0.6859(0.681119)|**2.0455278924.**|不使用POI数据|
|MLP      |0.7401|0.7091(0.70651)|**2.201941466.**|加入均值数据为53维|


### POI数值特征离散化
|模型/方法	|Train MAE|Dev MAE|Pubulic Test MAE|备注说明|
|---------|:---:|:----:|:--:|------|
|ETR      |0.0012|0.9691|**2.1448**|对离散型特征作One-Hot处理|
|MLP      |0.6640|0.6853|**2.058**|对结果四舍五入|

### 将状态为0或2的视为需求量加1，1视为需求量加0
|模型/方法	|Train MAE|Dev MAE|Pubulic Test MAE|备注说明|
|---------|:---:|:----:|:--:|------|
|MLP |0.5919|0.6121|**2.1574**|不使用ID|
|MLP |0.5845|0.6049|**2.1474**|使用ID|

## Copyright
The Owner of this project is Dr.E(Tongji University 436Lab), My contribute is the data process.
