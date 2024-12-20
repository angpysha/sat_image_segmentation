# Інформація про датасет

Для датасету було обрано файли з ресурсу [Sentinel hub](https://apps.sentinel-hub.com/eo-browser) формату Sentinel-2.

Обробка та підготовка файлів відбувається наступним чином:
1. Розмічення завантажених файлів за допомогою власної розробленої програми, яку можна скачати за [посиланням](https://install.appcenter.ms/users/andrew.petrowski/apps/sentinelimagecropper/distribution_groups/public%20testers).
2. В реузльтаті отримаємо файл **dataset.config** у форматі JSON, який містить інформацію про області та категорії

_Приклад:_

```{"categories":["sands"],"items":[{"category":"sands","width":853.0316824471959,"height":635.8517698470502,"top":0.0,"left":0.0}]}```

3. Подальша обрабка відбувається на Python

- Завантаження файлів за допомогою rasterio (вона по суті є оболонкою над GDAL)
- Вирізання зображення за допомогою вищезгаданого файлу **dataset.config**
- У результаті отримаємо, словник в якому ключ буде назва категорії, а значення - список вирізаних фрагментів зображення
- Далі відбувається операції конвертації двовимірних масивів в одновимірні, **тут важливо при операції reshape використовувати реальні розміри зображення (а не -1), оскільки тоді конвертація буде неправильною і при повернені назад зображення буде вже не таким**
- Після конвертації відбувається створення двох масиів numpy, один з яких містить в собі зображення, а інший - категорії (категорії вставляються по довжині списку зображень в словнику)
- В результаті отримаємо два масиви розмірністю (N,12) та (N,1)
- Далі ці дані можна подавати на відокремлення тестової та тренувальної вибірки

Завантажити його можна за посиланнями:
1. [Тренувальний](https://1drv.ms/u/s!AoIbr-LbEEDPl4gzP90htG7SbrtzAg)
2. [Верифікація](https://1drv.ms/u/s!AoIbr-LbEEDPl4lAoBqgACwFNLq6og?e=DKWVU3)