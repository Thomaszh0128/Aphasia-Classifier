import batchalign as ba  
nlp = ba.BatchalignPipeline.new("asr,speaker,morphosyntax", lang="eng", num_speakers=2)  
doc = ba.Document.new(media_path="/workspace/SH001/videos/ACWT07a.wav", lang="eng")  
doc = nlp(doc)  
chat = ba.CHATFile(doc=doc)  
chat.write("/workspace/SH001/vid_output/output.cha", write_wor=True)
