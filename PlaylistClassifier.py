#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


def get_playlist_tracks(username,playlist_id, sp):
    # SPOTIFY ONLY ALLOWS 100 TRACKS TO BE OBTAINED AT A TIME
    # LOOPS THROUGH A PLAYLISTS TRACKS AND OBTAINS ALL OF THEM
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


# In[3]:


def playlist_dict(username, sp):
    # CREATES A DICTIONARY OF A USER'S PLAYLISTS AND THEIR IDS
    playlists = {}
    for playlist in sp.user_playlists(username)['items']:
        playlists[playlist['id']] = playlist['name']
    return playlists


# In[4]:


def tracks_dict(username, playlists, sp):
    # CREATES A DICTIONARY WITH A USER'S PLAYLISTS AND ALL THEIR TRACKS
    tracks_dict = {}
    for playlist_id in playlists.keys():
        tracklist = get_playlist_tracks(username, playlist_id, sp)
        
        tracks = []
        for track in tracklist:
            if track['track'] is not None and track['track']['type'] == 'track':
                # IF THE TRACK IN THE PLAYLIST IS A PODCAST, SHOW, ETC, IT IS NOT ADDED
                tracks.append(track['track']['id'])
                
        tracks_dict[playlist_id] = tracks
    return tracks_dict


# In[5]:


def all_track_features(tracks_dictionary, sp):
    # OBTAINS THE AUDIO FEATURES OF ALL THE TRACKS 
    all_tracks = []
    for pl_id, tracklist in tracks_dictionary.items():
        playlist_tracks_feats = []
        for i in range(0, len(tracklist), 50):
            #sp.audio_features ONLY GETS 50 AT A TIME 
            audio_feat = sp.audio_features(tracklist[i:i+50])
            for item in audio_feat:
                if item is None:
                    audio_feat.remove(item)
                else:
                    item['Playlist'] = pl_id
            playlist_tracks_feats += audio_feat
        all_tracks += playlist_tracks_feats
    return all_tracks


# In[6]:


columns = ['danceability', 'energy', 'key', 
           'loudness', 'mode', 'speechiness', 
           'acousticness', 'instrumentalness', 'liveness',
          'valence', 'tempo', 'Playlist']

def create_dataframe(all_tracks, playlists):
    # BUILDS A DATAFRAME OF THE AUDIO FEATURES OF TRACKS 
    training_data = pd.DataFrame(all_tracks)
    training_data = training_data[columns]
    
    playlists_nominal = {}
    i = 1
    # CREATING A DICTIONARY OF TARGET VALUES ASSOCIATED WITH PLAYLISTS 
    for key in playlists.keys():
        playlists_nominal[key] = i
        i += 1
    
    def set_target(pl_col):
        return playlists_nominal[pl_col]
    
    training_data['target'] = training_data['Playlist'].apply(set_target)
    return training_data


# In[7]:


def classify_track(playlists, track_id, knn, sp):
    # RETURNS THE CLASSIFIED PLAYLIST THAT THE TRACK BELONGS IN
    features = sp.audio_features(track_id)
    feat_list = []
    for trait in columns[:-1]:
        feat_list.append(features[0][trait])
    
    result = knn.predict([feat_list])
    return playlists[result[0]]


# In[ ]:




