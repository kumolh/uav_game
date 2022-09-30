from PIL import Image
import PIL.ImageOps
import os

# path = 'imgs'
# files = os.listdir(path)
# for file in files:
#     if os.path.isdir(path + '/' + file):
#         for img_name in os.listdir(path + '/' + file):
#             img = Image.open(path + '/' + file + '/' + img_name)
#             if img.mode == 'RGBA':
#                 r,g,b,a = img.split()
#                 rgb_img = Image.merge('RGB', (r, g, b))
#                 inverted_img = PIL.ImageOps.invert(rgb_img)
#                 r2,g2,b2 = inverted_img.split()
#                 fin_img = Image.merge('RGBA', (r2,g2,b2,a))
#                 fin_img.save(path + '/' + file + '/new_' + img_name)
#             else:
#                 inverted_img = PIL.ImageOps.invert(img)
#                 inverted_img.save('imgs/trying.png')

img = Image.open('imgs/trans_summary.png')
if img.mode == 'RGBA':
    r,g,b,a = img.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    inverted_img = PIL.ImageOps.invert(rgb_img)
    r2,g2,b2 = inverted_img.split()
    fin_img = Image.merge('RGBA', (r2,g2,b2,a))
    fin_img.save('imgs/trans_summary_new.png')
else:
    inverted_img = PIL.ImageOps.invert(img)
    inverted_img.save('imgs/trans_summary_new.png')