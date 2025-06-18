from aiogram.types import update
from telegram import Update, InputMediaPhoto
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackContext, filters
import os
from model import photo_processing

API_TOKEN = 'ваш токен'
save_dir = 'ваш путь до папки сохранения'
result_dir = 'ваш путь до папки отправления'

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Данный бот создан для демонстрации возможностей модели.'
                                    '\nПожалуйста, загрузите фотографию брюшка тритона, и модель выведет вам 5 самых похожих изоражений')

async def handle_photo(update: Update, context: CallbackContext) -> None:

    photo = update.message.photo[-1]

    try:
        # Сохраняем фото
        aphoto = update.message.photo[-1]
        # Получаем объект файла
        photo_file = await photo.get_file()
        
        filename = "image.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Сохраняем фото
        await photo_file.download_to_drive(filepath)
        
        await update.message.reply_text(
            '✅ Ваша фотография была успешно сохранена!\n'
            '⌛ Пожалуйста, подождите, пока завершится работа алгоритма'
        )
        
        await photo_processing()
        
    except Exception as e:
        await update.message.reply_text(f'Произошла ошибка при сохранении фотографии: {str(e)}')
    
    try:
        # Формируем список путей к результатам
        result_files = [os.path.join(result_dir, f"top{i}.jpg") for i in range(1, 6)]
        
        # Проверяем существование файлов
        existing_files = [f for f in result_files if os.path.exists(f)]
        
        if not existing_files:
            await update.message.reply_text("❌ Результаты обработки не найдены")
            return
        
        with open('/home/yakshka/prod2/send_photo/res.txt', 'r') as f:
            res_str = f.read()
            
        # Отправляем медиагруппу
        media_group = []
        for i, filepath in enumerate(existing_files[:5]):
            with open(filepath, 'rb') as f:
                # Добавляем описание только к первому фото
                caption = f"Результаты обработки:\n\n{res_str}" if i == 0 else None
                media_group.append(InputMediaPhoto(media=f, caption=caption))
        
        await context.bot.send_media_group(
            chat_id=update.effective_chat.id,
            media=media_group
        )
        
        await update.message.reply_text("✅ Готово! Вот 5 самых похожих изображений")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка отправки: {str(e)}")
        
    

def main() -> None:
    application = ApplicationBuilder().token("7810246805:AAFUhEJL92DGNuDUBPPgro2kHPmnPv2tShY").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    application.run_polling()


if __name__ == '__main__':
    main()