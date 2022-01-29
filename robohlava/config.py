"""
CONFIGURATION FILE
"""
""" FLAGS   """
flag_arduino = False
flag_voice = True
flag_realsense = True

""" CONSTANTS   """
# Video
WIDTH = 640
HEIGHT = 480

# Number of frames to start tracking/untracking person after appear/disappear
num_person2track = 20
num_person2untrack = 20

# Person class properties
person_memory_length = 20


""" CLASSES and COLORS in BGR format for openCV  """

color_dict_en = {
        'person' : [0, 255, 255], 'bicycle': [238, 123, 158], 'car' : [24, 245, 217], 'motorbike' : [224, 119, 227],
        'aeroplane' : [154, 52, 104], 'bus' : [179, 50, 247], 'train' : [180, 164, 5], 'truck' : [82, 42, 106],
        'boat' : [201, 25, 52], 'traffic light' : [62, 17, 209], 'fire hydrant' : [60, 68, 169], 'stop sign' : [199, 113, 167],
        'parking meter' : [19, 71, 68], 'bench' : [161, 83, 182], 'bird' : [75, 6, 145], 'cat' : [100, 64, 151],
        'dog' : [156, 116, 171], 'horse' : [88, 9, 123], 'sheep' : [181, 86, 222], 'cow' : [116, 238, 87],'elephant' : [74, 90, 143],
        'bear' : [249, 157, 47], 'zebra' : [26, 101, 131], 'giraffe' : [195, 130, 181], 'backpack' : [242, 52, 233],
        'umbrella' : [131, 11, 189], 'handbag' : [221, 229, 176], 'tie' : [193, 56, 44], 'suitcase' : [139, 53, 137],
        'frisbee' : [102, 208, 40], 'skis' : [61, 50, 7], 'snowboard' : [65, 82, 186], 'sports ball' : [65, 82, 186],
        'kite' : [153, 254, 81],'baseball bat' : [233, 80, 195],'baseball glove' : [165, 179, 213],'skateboard' : [57, 65, 211],
        'surfboard' : [98, 255, 164],'tennis racket' : [205, 219, 146],'bottle' : [140, 138, 172],'wine glass' : [23, 53, 119],
        'cup' : [102, 215, 88],'fork' : [198, 204, 245],'knife' : [183, 132, 233],'spoon' : [14, 87, 125],
        'bowl' : [221, 43, 104],'banana' : [181, 215, 6],'apple' : [16, 139, 183],'sandwich' : [150, 136, 166],'orange' : [219, 144, 1],
        'broccoli' : [123, 226, 195],'carrot' : [230, 45, 209],'hot dog' : [252, 215, 56],'pizza' : [234, 170, 131],
        'donut' : [36, 208, 234],'cake' : [19, 24, 2],'chair' : [115, 184, 234],'sofa' : [125, 238, 12],
        'pottedplant' : [57, 226, 76],'bed' : [77, 31, 134],'diningtable' : [208, 202, 204],'toilet' : [208, 202, 204],
        'tvmonitor' : [208, 202, 204],'laptop' : [159, 149, 163],'mouse' : [148, 148, 87],'remote' : [171, 107, 183],
        'keyboard' : [33, 154, 135],'cell phone' : [206, 209, 108],'microwave' : [206, 209, 108],'oven' : [97, 246, 15],
        'toaster' : [147, 140, 184],'sink' : [157, 58, 24],'refrigerator' : [117, 145, 137],'book' : [155, 129, 244],
        'clock' : [53, 61, 6],'vase' : [145, 75, 152],'scissors' : [8, 140, 38],'teddy bear' : [37, 61, 220],
        'hair drier' : [129, 12, 229],'toothbrush' : [11, 126, 158]}

classes_en = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

color_dict_cz = {
        'človek': [0, 128, 255], 'bicykel': [238, 123, 158], 'auto': [24, 245, 217], 'motorka': [224, 119, 227],
        'lietadlo': [154, 52, 104], 'autobus': [179, 50, 247], 'vlak': [180, 164, 5], 'kamión': [82, 42, 106],
        'loď': [201, 25, 52], 'semafor': [62, 17, 209], 'hydrant': [60, 68, 169], 'stopka': [199, 113, 167],
        'parkovacie hodiny': [19, 71, 68], 'lavička': [161, 83, 182], 'vták': [75, 6, 145], 'mačka': [100, 64, 151],
        'pes': [156, 116, 171], 'kôň': [88, 9, 123], 'ovca': [181, 86, 222], 'krava': [116, 238, 87],
        'slon': [74, 90, 143], 'medveď': [249, 157, 47], 'zebra': [26, 101, 131], 'žirafa': [195, 130, 181],
        'batoh': [242, 52, 233], 'dáždnik': [131, 11, 189], 'kabelka': [221, 229, 176], 'kravata': [193, 56, 44],
        'batožina': [139, 53, 137], 'frisbee': [102, 208, 40], 'lyže': [61, 50, 7], 'snowboard': [65, 82, 186],
        'lopta': [65, 82, 186], 'šárkan': [153, 254, 81], 'baseballova pálka': [233, 80, 195],
        'baseballova rukavica': [165, 179, 213],'skateboard' : [57, 65, 211], 'surfboard': [98, 255, 164],
        'tenisová raketa': [205, 219, 146], 'fľaša': [0, 255, 255], 'vínový pohár': [23, 53, 119],
        'šálka': [102, 215, 88], 'vydlička': [198, 204, 245], 'nožík': [183, 132, 233], 'lyžica': [14, 87, 125],
        'miska': [221, 43, 104], 'banán': [181, 215, 6], 'jablko': [16, 139, 183], 'sendvič': [150, 136, 166],
        'pomaranč' : [219, 144, 1], 'brokolica': [123, 226, 195], 'mrkva': [230, 45, 209], 'hot dog': [252, 215, 56],
        'pizza': [234, 170, 131], 'šiška': [36, 208, 234], 'koláčik': [255, 0, 255], 'stolička': [204, 204, 0],
        'pohovka': [125, 238, 12], 'kvetináč': [57, 226, 76], 'posteľ': [77, 31, 134], 'stôl': [208, 202, 204],
        'toaleta': [208, 202, 204], 'tvmonitor': [208, 202, 204], 'počítač': [255, 128, 0], 'myš': [148, 148, 87],
        'ovládač': [171, 107, 183], 'klávesnica': [33, 154, 135], 'mobil': [206, 209, 108],
        'mikrovonka': [206, 209, 108], 'rúra': [97, 246, 15], 'toustovač': [147, 140, 184], 'umývadlo': [157, 58, 24],
        'chladnička': [117, 145, 137], 'kniha': [255, 51, 153], 'hodiny': [53, 61, 6],'váza': [145, 75, 152],
        'nožnice': [8, 140, 38], 'macko': [37, 61, 220], 'fén na vlasy': [129, 12, 229], 'kefka': [11, 126, 158]}

classes_cz = ['človek', 'bicykel', 'auto', 'motorka', 'lietadlo', 'autobus', 'vlak', 'kamión', 'loď',
              'semafor', 'hydrant', 'stopka', 'parkovacie hodiny', 'lavička', 'vták', 'mačka',
              'pes', 'kôň', 'ovca', 'krava', 'slon', 'medveď', 'zebra', 'žirafa', 'batoh',
              'dáždnik', 'kabelka', 'kravata', 'batožina', 'frisbee', 'lyže', 'snowboard', 'lopta',
              'šárkan', 'baseballova pálka', 'baseballova rukavica', 'skateboard', 'surfboard', 'tenisová raketa',
              'fľaša', 'vinový pohár', 'šálka', 'vydlička', 'nožík', 'lyžica', 'miska', 'banán', 'jablko',
              'sendvič', 'pomaranč', 'brokolica', 'mrkva', 'hot dog', 'pizza', 'šiška', 'koláčik',
              'stolička', 'pohovka', 'kvetináč', 'posteľ', 'stôl', 'toaleta', 'tvmonitor', 'počítač',
              'myš', 'ovládač', 'klávesnica', 'mobil', 'mikrovonka', 'rúra', 'toustovač', 'umývadlo', 'chladnička',
              'kniha', 'hodiny', 'váza', 'nožnice', 'macko', 'fén na vlasy', 'kefka']

# BGR colors
color_tracking_person = (255, 255, 0)
color_person = (255, 0, 0)

