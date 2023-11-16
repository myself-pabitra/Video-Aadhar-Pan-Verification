from fastapi import FastAPI, HTTPException,status
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np

app = FastAPI()

# Placeholder database (replace with a real database connection)
database = {
    1: {
        "user_image_path": "temp/Debmalya.jpg",
        "pan_image_path": "temp/Debmalya Pan.jpeg",
        "aadhar_image_path": "temp/Debmalya Aadhar.jpeg",
    },
    2: {
        "user_image_path": "temp/my Image.jpeg",
        "pan_image_path": "temp/MY Pan HQ.jpeg",
        "aadhar_image_path": "temp/My Aadhar.jpeg",
    },
    3: {
        "user_image_path": "temp/Pabitra1.png",
        "pan_image_path": "temp/MY Pan HQ.jpeg",
        "aadhar_image_path": "temp/My Aadhar.jpeg",
    },
    4: {
        "user_image_path": "temp/my Image.jpeg",
        "pan_image_path": "temp/Pabitra_Pan_LQ.jpeg",
        "aadhar_image_path": "temp/My Aadhar.jpeg",
    },
    5: {
        "user_image_path": "temp/Pabitra1.png",
        "pan_image_path": "temp/Debmalya Pan.jpeg",
        "aadhar_image_path": "temp/My Aadhar.jpeg",
    },
}

def get_user_paths(user_id):
    user_data = database.get(user_id)
    if user_data:
        return user_data
    else:
        raise HTTPException(status_code=404, detail="User not found")


def calculate_face_distance(encoding1, encoding2):
    if encoding1 is not None and encoding2 is not None:
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        percentage_match = (1 - distance) * 100
        return percentage_match
    return None


def process_image(file_path):
    image = face_recognition.load_image_file(file_path)
    face_encodings = face_recognition.face_encodings(image, num_jitters=30)
    if not face_encodings:
        # raise HTTPException(status_code=400, detail=f"No face encodings found in {file_path}. Please try To Upload Better Quality Image" )
        print(f"No encoding found {file_path}")
    else:
        return np.mean(face_encodings, axis=0)

@app.post("/compare_faces/{user_id}", status_code=200)
async def compare_faces(user_id: int):
    try:
        # Initialize response_data
        response_data = {
            "user_id": user_id,
        }

        # Get user paths from the database
        user_paths = get_user_paths(user_id)

        # Process images
        user_encoding = process_image(user_paths["user_image_path"])
        pan_encoding = process_image(user_paths["pan_image_path"])
        aadhar_encoding = process_image(user_paths["aadhar_image_path"])

        if user_encoding is not None :
            if pan_encoding is not None:
                user_vs_pan = calculate_face_distance(user_encoding, pan_encoding)
                response_data["percentage_match_user_vs_pan"] = user_vs_pan
                response_data["user_vs_pan"] = "Matched Successfully" if user_vs_pan > 50 else "Mismatched"
            else:
                response_data = {
                    "error" : "No face encoding found in user Pan Image. Plase re-upload another Image",
                    "response_code" : status.HTTP_400_BAD_REQUEST
                    }
        else:
            response_data = {
                "error" : "No face encoding found in user image. Plase re-upload another Image",
                "response_code" : status.HTTP_400_BAD_REQUEST
            }
        if user_encoding is not None:
            if aadhar_encoding is not None:
                user_vs_aadhar = calculate_face_distance(user_encoding, aadhar_encoding)
                response_data["percentage_match_user_vs_aadhar"] = user_vs_aadhar
                response_data["user_vs_aadhar"] = "Matched Successfully" if user_vs_aadhar > 50 else "Mismatched"
            else:
                response_data = {
                "error" : "No face encoding found in Aadhar image. Plase re-upload another Image",
                "response_code" : status.HTTP_400_BAD_REQUEST
                }
        else:
            response_data = {
                "error" : "No face encoding found in User image. Plase re-upload another Image",
                "response_code" : status.HTTP_400_BAD_REQUEST
            }
        if  pan_encoding is not None:
            if  aadhar_encoding is not None:
                pan_vs_aadhar = calculate_face_distance(pan_encoding, aadhar_encoding)
                response_data["percentage_match_pan_vs_aadhar"] = pan_vs_aadhar
                response_data["pan_vs_aadhar"] = "Matched Successfully" if pan_vs_aadhar > 50 else "Mismatched"
            else:
                response_data = {
                    "error" : "No face encoding found in Aadhar image. Plase re-upload another Image",
                    "response_code" : status.HTTP_400_BAD_REQUEST
                }
        else:
            response_data = {
                "error" : "No face encoding found in Pan image. Plase re-upload another Image",
                "response_code" : status.HTTP_400_BAD_REQUEST
            }

        return response_data

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
