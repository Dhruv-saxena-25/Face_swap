import logging
import os



logs_path = os.path.join(os.getcwd(), 'logs')

os.makedirs(logs_path, exist_ok=True)

logging.basicConfig(filename=os.path.join(logs_path, 'app.logs'),
        format='[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
)