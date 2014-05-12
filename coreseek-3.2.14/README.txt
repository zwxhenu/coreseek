最新使用文档，请查看：http://www.coreseek.cn/products/products-install/

目录说明：
csft-x.y.z：coreseek源代码
mmseg-i.j.k：mmseg源代码
testpack：测试配置和数据包

testpack测试说明：
目录说明：
api：api接口和测试脚本
etc：配置文件
etc/pysource：python数据源脚本
var：运行数据
var/data：索引文件
var/log：搜索日志
var/test：测试源数据

配置1：
	测试对象：xml数据源，中文分词与搜索
    对应配置：etc/csft.conf
	测试数据：var/test/test.xml
	PHP程序：api/test_coreseek.php
	在线说明：http://www.coreseek.cn/products/products-install/install_on_bsd_linux/

配置2：
	测试对象：xml数据源，单字切分与搜索
    对应配置：etc/csft_cjk.conf
	测试数据：var/test/test.xml
	PHP程序：api/test_coreseek.php
	在线说明：http://www.coreseek.cn/products-install/ngram_len_cjk/

配置3：
	测试对象：mysql数据源，中文分词与搜索
    对应配置：etc/csft_mysql.conf
	测试数据：var/test/documents.sql
	PHP程序：api/test_coreseek.php
	测试说明：请先将测试数据导入数据库，并设置好配置文件中的MySQL用户密码数据库
	在线说明：http://www.coreseek.cn/products-install/mysql/

配置4：
	测试对象：python数据源，中文分词与搜索
    对应配置：etc/csft_demo_python.conf
	数据脚本：etc/pysource/csft_demo/__init__.py
	PHP程序：api/test_coreseek.php
	测试说明：请先安装Python 2.6 (x86)
	在线说明：http://www.coreseek.cn/products-install/python/

配置5：
	测试对象：python+mssql数据源，中文分词与搜索
    对应配置：etc/csft_demo_python_pymssql.conf
	数据脚本：etc/pysource/csft_demo_pymssql/__init__.py
	PHP程序：api/test_coreseek.php
	测试说明：请先安装Python 2.6 (x86)、pymssql（py2.6）
	在线说明：http://www.coreseek.cn/products-install/python/

coreseek-4.y.z测试：
配置6：
	测试对象：RT实时索引，中文分词与搜索
    对应配置：etc/csft_rtindex.conf
    PHP程序：api/test_coreseek_rtindex.php
	在线说明：http://www.coreseek.cn/products-install/rt-indexes/
	
配置7：
	测试对象：RT实时索引，单字切分与搜索
    对应配置：etc/csft_rtindex_cjk.conf
    PHP程序：api/test_coreseek_rtindex.php
	在线说明：http://www.coreseek.cn/products-install/rt-indexes/
