import os
import pickle
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import googleapiclient.discovery

class scraper():
    def __init__(self, api_key):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        self.wdir = Path(__file__).resolve().parents[0]
        self.save_dir = self.wdir / "response.pickle"
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.DEVELOPER_KEY = api_key
        self.service = googleapiclient.discovery.build(self.api_service_name, self.api_version, developerKey = self.DEVELOPER_KEY)

    def get_id(self,url: str) -> str:
        """ 
        Returns youtube video id from url 
        https://stackoverflow.com/questions/45579306/get-youtube-video-url-or-youtube-video-id-from-a-string-using-regex
        """
        u_pars = urlparse(url)
        quer_v = parse_qs(u_pars.query).get('v')
        if quer_v:
            return quer_v[0]
        pth = u_pars.path.split('/')
        if pth:
            return pth[-1]

    def get_comments(self, video_url: str, maxResults: int = 100, save_to_disk: bool = False):
        """
        Fetches comments from youtube video using Youtube v3 Data API 
        """
        response = self.service.commentThreads().list(
            part='snippet',
            maxResults= maxResults,
            textFormat='plainText',
            order='time',
            videoId=self.get_id(video_url)
        ).execute()

        comments = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in response['items']]

        if save_to_disk:
            with open(self.save_dir, 'wb') as handle:
                pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
        return comments         


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=XE3krf3CQls&ab_channel=deeplizard"
    api_key = "AIzaSyC5HZxK4bznwBldhwF_gJXodOqYurYlFqI"
    
    s = scraper(api_key)
    print(s.get_comments(url,save_to_disk=True))
