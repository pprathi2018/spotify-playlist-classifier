# spotify-playlist-classifier

A simple playlist classifier that using the KNN Machine Learning technique to allocate a song to one of a user's saved playlists. The user is allowed to select which of their playlists to analyze, and search for any song on Spotify!

###### Development Steps

Obtains User Authorization through Client Credentials Flow, and uses a web access token to retrieve information about a user's saved music through the Spotify Web API. My Client Credentials are removed from the source file. 

The spotipy python package is utilized to gather playlist data, including their tracks and ids. The data is manipulated into various dictionaries to allow retrievability.

A Panda's dataframe is created where the rows are the various tracks, and the columns are their corresponding audio features, to utilize in the KNN classifier.

K-Nearest-Neighbour Machine Learning classification is done on the data, using the scikitlearn package. An accuracy score and a prediction model are obtained. 

A clean, and readable UI is created using the PySimpleGUI Python package. 
