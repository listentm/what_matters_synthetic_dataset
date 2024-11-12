import requests
import pdb
proxies={
    "http":"http://127.0.0.1:7891",
    "https":"http://127.0.0.1:7891"
}
url="http://en.wikipedia.org/wiki/Main_Page"
response=requests.get(url,proxies=proxies)
print(response.status_code)


#ssh -R 7891:localhost:7890 liyisen@222.20.126.192 -p 22
