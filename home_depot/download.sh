curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/train.csv.zip?sv=2012-02-12&se=2016-04-16T21%3A51%3A47Z&sr=b&sp=r&sig=2TZWLWNhVGSLnujGbMkhS1onSInHUypseCs%2FJiILjss%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data' -H 'Connection: keep-alive' --compressed  > train.csv.zip

curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/test.csv.zip?sv=2012-02-12&se=2016-04-16T21%3A52%3A14Z&sr=b&sp=r&sig=D3JAiNGwaZ1Xc9cfjz0lcJiNfn9jBUyp6kdIo%2Ffahew%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data' -H 'Connection: keep-alive' --compressed > test.csv.zip

curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/product_descriptions.csv.zip?sv=2012-02-12&se=2016-04-16T21%3A52%3A30Z&sr=b&sp=r&sig=m2ymLtrYXbcJbVbgl5A57d9vc0s2sXTFzDuUbs5nYzQ%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data' -H 'Connection: keep-alive' --compressed > product_descriptions.csv.zip

curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/attributes.csv.zip?sv=2012-02-12&se=2016-04-16T21%3A52%3A40Z&sr=b&sp=r&sig=a5NFn5%2BrMfFRUEhUrN8DADLZ7iAC3uOBW3jb51znlnI%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data' -H 'Connection: keep-alive' --compressed > attributes.csv.zip

curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4853/sample_submission.csv.zip?sv=2012-02-12&se=2016-04-16T22%3A03%3A59Z&sr=b&sp=r&sig=MSsBUKNO9bv%2Bf4RLKNq4sjV0a1Q5244YcDA7mUVNMjg%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,cs;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/home-depot-product-search-relevance/data' -H 'Connection: keep-alive' --compressed > sample_submission.csv.zip

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
