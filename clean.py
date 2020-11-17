import os
for file in os.listdir('./Training'):
    if file.endswith('.png'):
        os.remove('./Training/'+file)


