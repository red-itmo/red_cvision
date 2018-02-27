import cv2


def mouse_callback(event, x, y, flags, param):
    global selecting_region, region_tl, region_br, mouse_pos

    if event == cv2.EVENT_RBUTTONDOWN:
        selecting_region = True
        region_tl = (x, y)
        region_br = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
        if selecting_region:
            region_br = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        selecting_region = False
        selected = True
        (tlx, tly), (brx, bry) = sorted((region_tl, region_br))
        print('ROI    TL:({}, {})    BR:({}, {})'.format(tlx, tly, brx, bry))
        if brx > tlx and bry > tly:
            roi = image_base[int(tly/SCALE):int(bry/SCALE), int(tlx/SCALE):int(brx/SCALE)]
            cv2.imwrite('draft/' + roi_name+'.png', roi)
        else:
            print('[-] Possible empty ROI')



path = 'draft/'
# roi_images = ['F20_20_B', 'F20_20_G', 'S40_40_B', 'S40_40_G', 'M20_100', 'M20','M30', 'R20',
#                 'Bearing_Box', 'Bearing', 'Axis', 'Distance_Tube', 'Motor']

roi_images = ['1']
raw = '1.jpg'
roi_name = ''
selected = False

for rim in roi_images:

    selected = False
    selecting_region = False
    region_tl = (0, 0)
    region_br = (0, 0)
    roi_name = rim
    mouse_pos = (0, 0)

    # WINDOW
    cv2.namedWindow('MAIN')

    # MOUSE CALLBACK
    cv2.setMouseCallback('MAIN', mouse_callback)
    new = None
    SCALE = 0.2
    frame_big = cv2.imread(path + raw)
    w, h, _ = frame_big.shape
    frame = cv2.resize(frame_big, (int(h * SCALE), int(w * SCALE)))

    while True:
        image_base = frame_big.copy()
        image = frame.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.putText(image, 'select: {}'.format(roi_name), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
        cv2.putText(image, '{}, {}'.format(*mouse_pos), mouse_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 1)

        if selecting_region:
            new = cv2.rectangle(image_base, (int(region_tl[0]/SCALE), int(region_tl[1]/SCALE)),
                                    (int(region_br[0]/SCALE), int(region_br[1]/SCALE)), (250, 250, 250), 0)

        cv2.imshow('MAIN', image)

        if new:
            cv2.imshow('MAIN', new)

        if selected:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
