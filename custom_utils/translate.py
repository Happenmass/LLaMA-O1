from datasets import load_from_disk
from google.cloud import translate

ds = load_from_disk("/home/ecs-user/nas_training_data/OpenLongCoT_hfdata")

def translate_text(
    text: str = "YOUR_TEXT_TO_TRANSLATE", project_id: str = "adept-now-440806-b4"
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": "zh",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print(f"Translated text: {translation.translated_text}")

    return response

print(translate_text("Hello, world!"))