# 文件说明
- poi_process.py：将原始POI数据转换成区域id以及其POI值的共11维数据

		python poi_process.py

- poi_discretize.py:对poi中的数据进行离散化,以及统计7月份训练集中geo出现的次数

        python poi_discretize.py  

- poi_discretize.py:对poi中的数据进行聚类,聚类结果分成10类，结果保存为10维向量，所属类别为1，其余为0

        python poi_kmeans.py

- weather_process.py：将原始的weather数据转换成日期，时间，天气代码共3维数据

		python weather_process.py
		
- generate\_train\_dev\_input.py：用于将7月份（或8月份）的原始数据转换与公开测试集格式相同的输入文件，所产生的文件可用于生成神经网络模型的输入训练集（或开发集）

		python generate_train_dev_input.py ../raw_data/train_July.csv ../data/train_input.csv
		python generate_train_dev_input.py ../raw_data/train_Aug.csv ../data/dev_input.csv
		
- generate\_train\_dev.py：利用上一步生成的输入文件，生成最终的训练集（或开发集）

        原始POI数据
		python generate_train_dev.py ../data/train_input.csv ../data/train.csv
		python generate_train_dev.py ../data/dev_input.csv ../data/dev.csv
		
	    ##注意要去修改读POI的文件
		discretize后的POI数据
		python generate_train_dev.py ../data/train_input.csv ../data/poi_discretize/train.csv
		python generate_train_dev.py ../data/dev_input.csv ../data/poi_discretize/dev.csv
		
		##注意要去修改读POI的文件
		kmeans后的POI数据
		python generate_train_dev.py ../data/train_input.csv ../data/poi_kmeans/train.csv
		python generate_train_dev.py ../data/dev_input.csv ../data/poi_kmeans/dev.csv

- generate_test.py：利用提供的公开测试集（或未来提供的最终测试集），生成用于模型输入的测试集数据（或最终测试集数据）

		python generate_test.py ../raw_data/test_id_Aug_agg_public5k.csv ../data/test_public.csv
        python generate_test.py ../raw_data/test_id_Aug_agg_private5k.csv ../data/test_private.csv
        
        ##注意要去修改读POI的文件
        discretize后的POI数据
        python generate_test.py ../raw_data/test_id_Aug_agg_public5k.csv ../data/poi_discretize/test_public.csv

        ##注意要去修改读POI的文件
        kmeans后的POI数据
        python generate_test.py ../raw_data/test_id_Aug_agg_public5k.csv ../data/poi_kmeans/test_public.csv


- feature_process.py 加上均值特征之后产生训练集和测试集数据
        python feature_process.py train.csv train_53.csv
        python feature_process.py test_public.csv test_public_52.csv
        python feature_process.py dev.csv dev_53.csv
        python feature_process.py test_private.csv test_private_52.csv



