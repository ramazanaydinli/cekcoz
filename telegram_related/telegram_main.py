import os.path
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import numpy as np
from datetime import datetime
import cv2 as cv
from cekcoz_object_detection import object_detection_main

# Bot tokenını buraya ekleyin
TOKEN = "6517406670:AAF6VHk1RjlVa5FzAxW_ZDmLXQtlJZwatlU"


# Resim mesajı alındığında ekranda gözükecek yazı aşağıdadır.
RESPONSE_MESSAGE = 'Teşekkür ederim, resmi aldım. Birazdan cevabı ileteceğim.'


def start(update):
    user_id = update.message.from_user.id
    welcome_message = f"ÇekÇöze Hoşgeldin, {update.message.from_user.first_name} ({user_id})!"
    update.message.reply_text(welcome_message)


def reply_to_image(update, context):
    # print("Resim mesajı alındı!")  # Bu mesajı konsolda görmelisiniz
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
    filename = f"{user_name.replace(' ', '_')}_{current_date}_{current_time.replace(':', '-')}.jpg"
    # Byte dizisini bir numpy dizisine dönüştür
    nparr = np.frombuffer(file, np.uint8)

    if message.photo:
        context.bot.send_message(chat_id=chat_id, text=RESPONSE_MESSAGE)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        saving_path = os.path.join(os.getcwd(), "gelen_resimler", filename)
        cv.imwrite(saving_path, img)
        new_image_path = object_detection_main.object_detection(img, saving_path, filename)
        with open(new_image_path, 'rb') as img_file:
            context.bot.send_photo(chat_id=chat_id, photo=img_file, caption= "Al yarramın başı bak gördün mü cevap")

def hello(update):
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
