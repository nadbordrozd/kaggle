curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/train.csv.zip?sv=2012-02-12&se=2016-02-19T20%3A40%3A26Z&sr=b&sp=r&sig=MFLrDhWmZg4%2B6%2FRGQpa7I1iQ8ifLERkCqlHdPIc9iVk%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data?train.csv.zip' -H 'Connection: keep-alive' --compressed > train.csv.zip
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/test.csv.zip?sv=2012-02-12&se=2016-02-19T20%3A46%3A12Z&sr=b&sp=r&sig=O9NMRbe6Ej9Pl4yixJcbzj43Homk%2BLWGwht9HeuQUH4%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data?train.csv.zip' -H 'Connection: keep-alive' --compressed > test.csv.zip
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/product_descriptions.csv.zip?sv=2012-02-12&se=2016-02-19T20%3A46%3A33Z&sr=b&sp=r&sig=djz2X1TIcTFZ2kUnkrXUnd355x4vLLS%2BRNHCKEwXTLk%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data?train.csv.zip' -H 'Connection: keep-alive' --compressed > product_descriptions.csv.zip
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/attributes.csv.zip?sv=2012-02-12&se=2016-02-19T21%3A16%3A03Z&sr=b&sp=r&sig=CYIPbuytMQ%2FyqpzLlnFWnuuLuO%2Fyw8qq49Md%2Fqo9OD4%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data?train.csv.zip' -H 'Connection: keep-alive' --compressed > attributes.csv.zip
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/sample_submission.csv.zip?sv=2012-02-12&se=2016-02-19T21%3A18%3A10Z&sr=b&sp=r&sig=Q9SmfItDQivk5r26JNqD3Ggqc8KpBDliSgxcrhfiDn0%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data?train.csv.zip' -H 'Connection: keep-alive' --compressed > sample_submission.csv.zip

unzip train.csv.zip
unzip test.csv.zip
unzip product_descriptions.csv.zip
unzip attributes.csv.zip
unzip sample_submission.csv.zip

rm train.csv.zip
rm test.csv.zip
rm product_descriptions.csv.zip
rm attributes.csv.zip
rm sample_submission.csv.zip

mkdir data
mv *.csv data/
