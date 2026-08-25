from tacet.serve.settings import load_settings, Settings
import os

os.environ['TACET_CORS_ORIGINS'] = '*'

settings = load_settings()

print(settings.cors_origins)
