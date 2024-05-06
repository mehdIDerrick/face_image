from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import face_recognition
import os
import uvicorn
app = FastAPI()
@app.post("/compare_faces/")
async def compare_faces(known_image: UploadFile = File(...), unknown_image: UploadFile = File(...)):
    try:
        # Créer des fichiers temporaires pour les images
        with tempfile.NamedTemporaryFile(delete=False) as known_file, tempfile.NamedTemporaryFile(delete=False) as unknown_file:
            shutil.copyfileobj(known_image.file, known_file)
            shutil.copyfileobj(unknown_image.file, unknown_file)
            known_file_path = known_file.name
            unknown_file_path = unknown_file.name
        
        # Charger les images
        known_image = face_recognition.load_image_file(known_file_path)
        unknown_image = face_recognition.load_image_file(unknown_file_path)
        
        # Encoder les visages
        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        
        # Comparer les visages
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        
        # Supprimer les fichiers temporaires
        os.unlink(known_file_path)
        os.unlink(unknown_file_path)
        
        # Retourner le résultat de la comparaison
        if results[0]:
            return {"message": "Les visages correspondent."}
        else:
            return {"message": "Les visages ne correspondent pas."}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)