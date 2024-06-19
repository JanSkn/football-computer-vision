from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self) -> None:
        self.team_colours = {}   # key: team_id (1 or 2), value: colour
        self.player_team_dict = {}      # key: player_id, value: team_id

    def get_clusters(self, image: np.ndarray) -> KMeans:
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0).fit(image_2d)

        return kmeans

    def get_player_colour(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2), :]  # only select shirt

        kmeans = self.get_clusters(top_half_image)
        labels = kmeans.labels_ # 0 or 1

        # reshape labels back to image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        # look at corner points to check which cluster is the background
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        background_cluster = max(corner_clusters, key=corner_clusters.count)    # cluster with most appearances in the corners (assumption that corners are background)
        player_cluster = 1 - background_cluster

        player_colour = kmeans.cluster_centers_[player_cluster] # "mean" colour of the player's shirt cluster

        return player_colour

    def assign_team_colour(self, frame: np.ndarray, tracks: Dict[str, List[Dict]]) -> None:
        player_colours = []

        # get shirt colour of the players in the frame
        for _, player in tracks.items():
            bbox = player["bbox"]
            player_colour = self.get_player_colour(frame, bbox)
            player_colours.append(player_colour)

        # of all slightly different colours, determine the 2 "mean" colours as team colours
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0, n_init=10).fit(player_colours)

        self.kmeans = kmeans

        # team with ID 1 and team with ID 2 and their colour
        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: np.ndarray, bbox: List[float], player_id: int) -> int:
        # assign each player a team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_colour = self.get_player_colour(frame, bbox)

        # reshape colour from 3 x 1 to 1 x 3 for prediction
        team_id = self.kmeans.predict(player_colour.reshape(1, -1))[0] + 1      # [0], because player_colour shape is [[R G B]] + 1, because team_id 1 or 2, not 0 or 1

        self.player_team_dict[player_id] = team_id
        
        return team_id
    
    def get_teams(self, frames: np.ndarray, tracks: Dict[str, List[Dict]]) -> None:
        """
        Called in main.
        Adds keys to tracks.
        """
        # only assign teams if players found/player tracking selected
        if len(tracks["players"][0]) > 0:  # default empty: [{}]
            self.assign_team_colour(frames[0], tracks["players"][0])

            for frame_num, player in enumerate(tracks["players"]):
                for player_id, player_track in player.items():
                    team = self.get_player_team(frames[frame_num], player_track["bbox"], player_id)

                    # add new keys team and team_colour
                    tracks["players"][frame_num][player_id]["team"] = team
                    tracks["players"][frame_num][player_id]["team_colour"] = self.team_colours[team]
