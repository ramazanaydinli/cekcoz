import os.path

import telegram.ext
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from object_detection_main import object_detection
import numpy as np
from datetime import datetime
import cv2 as cv

# Bot tokenını buraya ekleyin
TOKEN = "6517406670:AAF6VHk1RjlVa5FzAxW_ZDmLXQtlJZwatlU"

RESPONSE_MESSAGE = 'Teşekkür ederim! Resmi aldım.'

def start(update, context):
    user_id = update.message.from_user.id
    welcome_message = f"ÇekÇöze Hoşgeldiniz, {update.message.from_user.first_name} ({user_id})! İşlem kategorileri hakkında bilgi almak için /kategoriler komutunu kullanabilirsiniz."
    update.message.reply_text(welcome_message)

def reply_to_image(update, context):
    print("Resim mesajı alındı!")  # Bu mesajı konsolda görmelisiniz
    message = update.message
    user = message.from_user
    chat_id = message.chat_id

    # Resmin dosya kimliğini al
    file_id = message.photo[-1].file_id
    file_info = context.bot.get_file(file_id)

    # Resmi indir
    file = file_info.download_as_bytearray()
    user_name = user.first_name + " " + (user.last_name if user.last_name else "")
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    text = f"User: {user_name}\nTime: {current_time}\nDate: {current_date}"
    filename = f"{user_name.replace(' ', '_')}_{current_date}_{current_time.replace(':', '-')}.jpg"
    # Byte dizisini bir numpy dizisine dönüştür
    nparr = np.frombuffer(file, np.uint8)


    if message.photo:
        context.bot.send_message(chat_id=chat_id, text=RESPONSE_MESSAGE)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        saving_path = os.path.join("gelen_resimler", filename)
        cv.imwrite(saving_path, img)
        object_detection(img)
def hello(update, context):
    update.message.reply_text("Merhaba Dünya!")



def main():
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # "/start" komutunu işlemek için bir komut işleyici ekleyin
    dispatcher.add_handler(CommandHandler("start", start))

    # Gruba gönderilen resimlere cevap vermek için bir mesaj işleyici ekleyin
    dispatcher.add_handler(MessageHandler(Filters.photo, reply_to_image))

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, hello))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()