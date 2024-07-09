import os
import pandas as pd


if __name__ == '__main__': 
	
	WINDOW = 1000
	corpus_name = 'GuitarSet_test'
	songs_list = os.listdir(f'./{corpus_name}/features')
	songs_list.remove('all-files-in-corpus')
	os.makedirs(f'./{corpus_name}/features/all-files-in-corpus-rolling-avg', exist_ok=True)

	for song_name in songs_list:
		print(song_name)
		path_single = f'./{corpus_name}/features/{song_name}/{song_name}.csv'

		df = pd.read_csv(path_single, index_col=['Unnamed: 0'])
		df_rolling = df.rolling(WINDOW).mean()
		df_rolling = df_rolling.dropna()
		df_rolling.to_csv(f'./{corpus_name}/features/{song_name}/{song_name}_1000rolling.csv', index=True)  

		path_all = f'./{corpus_name}/features/all-files-in-corpus/{song_name}.csv'
		df = pd.read_csv(path_all, index_col=['Unnamed: 0'])
		df_rolling = df.rolling(WINDOW).mean()
		df_rolling = df_rolling.dropna()
		df_rolling.to_csv(f'./{corpus_name}/features/all-files-in-corpus-rolling-avg/{song_name}_1000rolling.csv', index=True)  
