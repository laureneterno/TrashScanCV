import boto3
import os
import cv2
import numpy as np
import tensorflow as tf
import json

class_mapping = {
    0: "Battery",
    1: "Biological",
    2: "Cardboard",
    3: "Glass",
    4: "Metal",
    5: "Paper",
    6: "Plastic",
    7: "Trash"
}

def preprocess_image(image):
    """
    Normalize the image by subtracting the mean and dividing by the standard deviation.
    """
    image = np.array(image, dtype=np.float32)
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + 1e-7)

def preprocess_image_s3(bucket_name, object_key, s3_client):
    """
    Fetch and preprocess an image from S3.
    """
    # Fetch image from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    image_data = response['Body'].read()

    # Decode and preprocess the image
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = preprocess_image(image)       # Normalize the image

    return np.expand_dims(image, axis=0)


def lambda_handler(event, context):
    try:
        # validate s3 uri parameter 
        if 'queryStringParameters' not in event or 'path' not in event['queryStringParameters']:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': "Missing 'path' query parameter."})
            }

        path = event['queryStringParameters']['path']
        if not path.startswith("s3://"):
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': "Invalid path: expected an S3 URI."})
            }

        # parse the uri 
        bucket_name, object_key = path.replace("s3://", "").split("/", 1)

        s3_client = boto3.client('s3')
        input_data = preprocess_image_s3(bucket_name, object_key, s3_client)

        # load the model (to not cache)
        MODEL_PATH = '/opt/model/saved_model'
        model = tf.saved_model.load(MODEL_PATH)
        inference_fn = model.signatures['serving_default']

        # predict object type 
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        predictions = inference_fn(input_tensor)
        prediction_array = list(predictions.values())[0].numpy()
        predicted_label = class_mapping.get(int(np.argmax(prediction_array, axis=1)[0]))

        # output response 
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'predicted_label': predicted_label})
        }

    except Exception as e:
        # failed response 
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }