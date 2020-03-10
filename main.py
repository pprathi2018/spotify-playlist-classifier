#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PlaylistClassifier
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd

# SETTING UP USER CREDENTIALS AND LOADING A TOKEN

cid = '4485524162d84a799f2a1258d16cde82'
secret = '812fc20998f7469e9da6aa5080a8d9cf'
username = '9zanehtcggtsl6ps7pcdb99ih'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# PLAYLIST-READ-PRIVATE ALLOWS ACCESS TO A USER'S PLAYLIST
scope = 'user-library-read playlist-read-private'

token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                  redirect_uri='http://localhost:8080/callback')

if token:
    sp = spotipy.Spotify(auth=token)
else:
    token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                  redirect_uri='http://localhost:8080/callback')


# In[2]:


playlists = PlaylistClassifier.playlist_dict(username, sp)
playlists


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# KNN MACHINE LEARNING CLASSIFIER
def perform_knn(training_data):
    features = training_data.iloc[:, :-2]
    target = training_data.iloc[:, -2]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 10)  

    # Feature scaling to normalize weightage of each feature 
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(10)
    knn.fit(X_train, y_train)
    
    return knn, knn.score(X_test, y_test)


# In[4]:


import PySimpleGUI as sg
def make_checkbox(playlists):
    final_list = []
    for i in range(0, len(playlists.values()), 3):
        three_item_list = []
        for name in list(playlists.values())[i:i+3]:
            three_item_list.append(sg.Checkbox(name, size=(10, 1), key=name))
        final_list.append(three_item_list)
    return final_list

make_checkbox(playlists)


# In[6]:


# CREATING A USER INTERFACE USING PYSIMPLEGUI
import PySimpleGUI as sg

sg.theme('Dark')

# MAKING THE CHECKBOXES FOR UI:
def make_checkbox(playlists):
    final_list = []
    for i in range(0, len(playlists.values()), 3):
        three_item_list = []
        for name in list(playlists.values())[i:i+3]:
            three_item_list.append(sg.Checkbox(name, size=(12, 1), key=name))
        final_list.append(three_item_list)
    return final_list
        

layout = [
    [sg.Text('Playlist Classifier App', size=(30, 1), justification='center', 
             font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
    [sg.Text('Select your playlists: ')],
    [sg.Frame(layout = make_checkbox(playlists), title='Your Playlists', title_color='dark green', relief=sg.RELIEF_SUNKEN)],
    [sg.Button('Submit'), sg.Button('Quit')],
    [sg.Text('')],
    [sg.Text('Search for a song: ')],
    [sg.Input(key='search'), sg.Button('Search')],
    [sg.Listbox(values=(), key='suggestions', size=(40,3)), sg.Button('Classify')],
    [sg.Text('')],
    [sg.Text('', key='answer', size=(40, 1))],
    [sg.Text('_' * 70)],
    [sg.Multiline(key='accuracy_score')]  
]

window = sg.Window('Playlist Classifier!', layout)

while True:
    event, values = window.read()
    
    def read_selections(playlists):
        # READS WHICH CHECKBOXES ARE SELECTED
        wanted_playlists = {}
        for pl_id, name in playlists.items():
            if values[name]:
                wanted_playlists[pl_id] = name
        return wanted_playlists
    
    def create_search_list(search):
        # CREATES A LIST OF OPTIONS FOR A SEARCHED TRACK
        # PRESENTS EACH SEARCH WITH THE TRACK NAME AND ARTIST 
        searched_list = {}
        elements = sp.search(search, limit=5, type='track')['tracks']['items']
        for suggestion in elements:
            searched_list[suggestion['name'] + ' - ' + suggestion['artists'][0]['name']] = suggestion['id']
        return searched_list, tuple(searched_list.keys())
    
    if event == 'Submit':
        wanted_playlists = read_selections(playlists)
        clicked_submit = True
        
        if not wanted_playlists:
            window['accuracy_score'].update('Please select at least 1 playlist')
        else:
            tracks_dict = PlaylistClassifier.tracks_dict(username, wanted_playlists, sp)
            all_tracks = PlaylistClassifier.all_track_features(tracks_dict, sp)
            training_data = PlaylistClassifier.create_dataframe(all_tracks, playlists)
        
    if event in(None, 'Quit'):
        break
    
    if event == "Search":
        if values['search'] is not '':
            searched_track = create_search_list(values['search'])
            window['suggestions'].update(searched_track[1])
        else:
            window['suggestions'].update('')
            
    
    if values['search'] is not '' and event == 'Classify' is not None and clicked_submit:
        if not read_selections(playlists):
            window['answer'].update('Please select at least 1 playlist')
        else:
            knn = perform_knn(training_data)
            accuracy_text = 'The accuracy of the classification is ' + str(format(knn[1]*100, '0.1f')) + '%'
            window['accuracy_score'].update(accuracy_text)
            
            print(values['suggestions'])
            if len(values['suggestions']) != 0:
                track_id = searched_track[0][values['suggestions'][0]]
                print([values['suggestions'][0]])
                answer = PlaylistClassifier.classify_track(playlists, track_id, knn[0], sp)
                window['answer'].update('Suggested Playlist: ' + answer)

    
window.close()


# In[ ]:




