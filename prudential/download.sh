curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4699/train.csv.zip?sv=2012-02-12&se=2016-02-06T18%3A46%3A07Z&sr=b&sp=r&sig=Nlo5qIxqxmV69x2OEIVo18O80voCBpBF%2BijJQ92vgDM%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,pl;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.97 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/prudential-life-insurance-assessment/data' -H 'Connection: keep-alive' --compressed > train.csv.zip
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4699/test.csv.zip?sv=2012-02-12&se=2016-02-06T18%3A48%3A08Z&sr=b&sp=r&sig=%2FHu9z6Z9G1v5XfIj5WOP%2B2BmE6wckmh%2FfYVww3vaCZk%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,pl;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.97 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/prudential-life-insurance-assessment/data' -H 'Connection: keep-alive' --compressed > test.csv.zip
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4699/sample_submission.csv.zip?sv=2012-02-12&se=2016-02-06T18%3A48%3A34Z&sr=b&sp=r&sig=43TpzaVM%2B0ZFyMJ3yn1vaubr7xf%2Bvk7uNAzkAvb%2Bcqw%3D' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-US,en;q=0.8,pl;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.97 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/prudential-life-insurance-assessment/data' -H 'Connection: keep-alive' --compressed > sample_submission.csv.zip
unzip train.csv.zip
unzip test.csv.zip
unzip sample_submission.csv.zip

rm train.csv.zip
rm test.csv.zip
rm sample_submission.csv.zip
