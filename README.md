Hello, welcome to my ETL pipeline for preparing sound files for TTS model training. 

This pipeline takes wave files as input, but could easily be reconfigured for other formats.

_NB: I had to download and install ffmpeg in order to use pydub, though this was pretty simple to do._


#### process_audio.py - Outline
- Before the pipeline itself, we have the imports, and various constants and parameters that can be altered
1. The create_db function creates a database in which we will store:
    - ID (auto-incrementing primary key)
    - Source file name
    - File path for the audio segment
    - The transcribed text
    - The RMS (root mean square) of the audio file - average volume of file, this is used in the quality filters
    - A clipping metric - showing how much sound clipping occurs 
    - Music ratio - ratio between non-speech- and speech-band sound, used in quality filters to eliminate music
    - Overlap flag - shows whether there is overlap between a previous transcription, meaning a word might have been cut off
2. The create_batch function loads the files into a batch, as well as normalising them using pydub
3. The run_segmentation function segments each file into sections of an appropriate length, aiming to segment them 
  at a natural point using various pydub features
4. The run_audio_quality_filter function runs some quality checks on each audio file, dropping any sections that can't be used by the TTS model
5. The run_asr function runs the whisper-medium ASR model from OpenAI on each audio snippet, which transcribes any speech 
6. The run_text_quality_filter function will check for hallucinated or empty transcriptions, as well as generating the overlap flag
7. The save_processed_audio function saves the processed audio to a folder
8. The add_batch_metadata_to_db function then adds all the stored information into our SQL table
9. Finally, run_pipeline will run all those functions.

The output should be segmented audio files in a folder called processed_audio, and an SQL table populated with metadata for each of the segments. 

Some design notes:
- I found that the hardest part was definitely getting the right parameters for filtering out music, without also being sensitive to sound like birdsong and low background noises. As such the parameters I specify might be slightly overfit to these files, and it might be that in other usecases you would have to tinker with these. Overall, I found these values to be the best balance between filtering out non-speech audio and yielding a good proportion of quality sound files.
- Audio and text quality filters were separated out and placed either side of the ASR function to save compute on running the model.
- Some quality metrics were added to the metadata in case these need to be used downstream by AI engineers. For instance, if overlapping data is deemed useless then those files can easily be eliminated from the dataset.

I really enjoyed making this, especially learning about the libraries and models required - I had no idea you could do all these things with audio files in Python!

🖊️ Oscar Hill, completed 22-10-2025