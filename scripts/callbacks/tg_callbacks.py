import telegram_send
import gc


def send_tg_update(epoch, logs):
    # print(epoch)
    # print(log)
    message = f"Epoch: <i>{epoch}</i>\n\n"
    for key in logs.keys():
        message += f"> {key}: {logs[key]}\n"
    telegram_send.send(messages=[message], silent=True, parse_mode='HTML')


def send_training_start_update(logs):
    message = f"<b><i>Training started!</i></b>\n"
    for key in logs.keys():
        message += f"> {key}: {logs[key]}\n"
    telegram_send.send(messages=[message], silent=True, parse_mode='HTML')


def send_training_end_update(logs):
    message = u'\U00002714' + f"<i>Training end</i>\n"
    for key in logs.keys():
        message += f"> {key}: {logs[key]}\n"
    telegram_send.send(messages=[message], silent=True, parse_mode='HTML')


def clear_ram(epoch, logs):
    gc.collect()