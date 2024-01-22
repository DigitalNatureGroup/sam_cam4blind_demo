import json

import cv2 as cv
import numpy as np
import pygame
import pygame.locals as pygL
from typing import Tuple

from read_json import load_annotation
from gtts import gTTS
from io import BytesIO

import datetime
from GPT_utils_231230 import get_description, get_GPT4v_description_obj, invoke_GPT4V_w_image

from utils.localize import localStrFactory, supportedLocals, supportedLocals_fullname

import argparse

parser = argparse.ArgumentParser(description='')


parser.add_argument('--img', type=str, required = True, help='Image to view.')
parser.add_argument('--seg', type=str, required = True, help='Pre-made segmentation of the image.')

parser.add_argument('--lang', type=str, default='en', help='Language of the voice prompt, default: English')

args = parser.parse_args()

if args.lang in list(supportedLocals.__members__):
    LANG = supportedLocals[args.lang]
else:
    print(
        f"specified language {args.lang} not supported, supported languages are: {'/'.join(list(supportedLocals.__members__))}")
    exit(1)

L = localStrFactory(LANG.name)


img_path = args.img
json_path = args.seg
# config_path = args.conf
#img_path = 'pic_json_pair/example/a.jpg'
#json_path = 'pic_json_pair/example/seg_out.json'

pygame.mixer.init()
end_sound = pygame.mixer.Sound(r"notifications/Rhodes.ogg")
partial_end_sound = pygame.mixer.Sound(r"notifications/Blip.ogg")
out_of_frame_sound = pygame.mixer.Sound(r"notifications/Positive.ogg")
touch_playback_sound_path = r"bleep-41488.mp3"

color_inner = (255, 153, 255)
radius_inner = 5
thickness_inner = 2

color_outer = (255, 51, 51)
radius_outer = 7
thickness_outer = 3

JOYSTICK_DEADZONE = 0.1
JOYSTICK_SPEED = 3

MODE_FULLVIEW = 0
MODE_SEPARATE = 1
MODE_TOUCH = 2


def quick_speak(s: str):
    tts = gTTS(text=s, lang=LANG.name)

    with BytesIO() as fp:
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(5)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()


def get_list_description_GPT(focus_annot_index: int, len_annots_current: int, class_name: str,
                             class_proposals: list):  # , relative:Tuple[int, int]):
    descri_dict = {
        'target': [
            {
                "type": class_name,
                "description": class_proposals,
                "relative": ""
            },
        ],
        'surround': []
    }
    # quick_speak(L.get_str("please_wait"))
    return L.get_str("separate_content_f").format(nth=focus_annot_index +1 ,
                                                  num_all=len_annots_current,
                                                  )


def get_obj_count_str(len_annots_current: int):  # , relative:Tuple[int, int]):

    return L.get_str("separate_mode_count_f").format(
                                                  num_all=len_annots_current,
                                                  )


def openup_description(desr: str) -> str:
    try:
        dic = json.loads(desr)

        summary = dic["desc"]
        res = f" {summary} "
        if len(dic["surrounding"]) > 0:
            surrounding_s = dic["surrounding"]
            res += L.get_str("surround_f").format(surrounding_s=surrounding_s)
    except KeyError as e:
        print("KeyError")
        print(desr)
        print(e)

        return L.get_str("invoke_fail")
    else:
        return res

def openup_description_GPT4V(desr_json: str) -> str:
    try:
        dic = json.loads(desr_json)

        summary = dic["desc"]
        surrounding_s = dic["surrounding"]

        res = L.get_str("desc_surround_f").format(target_s=summary,surrounding_s=surrounding_s)
    except KeyError as e:
        print("KeyError")
        print(desr_json)
        print(e)

        return L.get_str("invoke_fail")
    else:
        return res


def make_partial_end_sound():
    pygame.mixer.music.stop()
    pygame.mixer.Sound.play(partial_end_sound)
    pygame.mixer.music.stop()


def make_end_sound():
    pygame.mixer.music.stop()
    pygame.mixer.Sound.play(end_sound)
    pygame.mixer.music.stop()


def is_in_bbox(cursor, bbox):
    x_cur, y_cur = cursor
    x, y, w, h = bbox
    return (x <= x_cur <= x + w) and (y <= y_cur <= y + h)


def is_on_mask(cursor, bbox, mask):
    x_cur, y_cur = cursor
    x, y, w, h = bbox
    if not 0 <= y_cur - y <= h:
        return None
    if not 0 <= x_cur - x <= w:
        return None
    # mask is y 1st, x 2nd
    return mask[y_cur - y][x_cur - x]


def on_mask_edge(cursor, bbox, mask):
    x_cur, y_cur = cursor
    x, y, w, h = bbox
    # mask is y 1st, x 2nd
    return (mask[y_cur - y][x_cur - x]),


def draw_cursor(img, x, y):
    cv.circle(img, (x, y), radius_inner, color_inner, thickness_inner)
    cv.circle(img, (x, y), radius_outer, color_outer, thickness_outer)


def convert_opencv_img_to_pygame(opencv_image):
    """
    OpenCVの画像をPygame用に変換.

    see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
    """
    opencv_image = opencv_image[:, :, ::-1]  # OpenCVはBGR、pygameはRGBなので変換してやる必要がある。
    shape = opencv_image.shape[1::-1]  # OpenCVは(高さ, 幅, 色数)、pygameは(幅, 高さ)なのでこれも変換。
    pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

    return pygame_image


sgn = lambda val: (1.0 if val > 0 else -1.0) if abs(val) > JOYSTICK_DEADZONE else 0


def draw_fullimg_mask(full_img: np.ndarray, crop_box, mask, color=(0, 0, 255)):
    y_size, x_size, channel = full_img.shape
    assert channel == 3
    x, y, w, h = crop_box
    # mask is y 1st, x 2nd
    full_mask = np.zeros([y_size, x_size], dtype=np.uint8)
    blank_img = np.zeros_like(full_img)

    full_mask[y:y + h, x:x + w] = mask

    masked_img = np.where(full_mask[..., None], np.asarray(color), blank_img)

    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    new_img = cv.addWeighted(full_img, 0.8, masked_img.astype(np.uint8), 0.2, 0)

    return new_img, full_mask
def draw_fullimg_mask_edge(full_img:np.ndarray, crop_box, mask, color = (0, 0, 255), mask_ratio : float = 0.2):
    y_size, x_size, channel = full_img.shape
    assert channel == 3
    assert 0.0 <= mask_ratio <= 1.0
    x, y, w, h = crop_box
    # mask is y 1st, x 2nd
    full_mask = np.zeros( [y_size,x_size],dtype=np.uint8 )
    blank_img = np.zeros_like(full_img)

    full_mask[y:y + h, x:x + w] = mask

    masked_img = np.where(full_mask[...,None], np.asarray((255,255,255)), blank_img)
    masked_img_bi = cv.cvtColor(masked_img.astype(np.float32), cv.COLOR_BGR2GRAY)
    edge_im = cv.Canny(masked_img_bi.astype(np.uint8),50,300)


    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    #new_img = cv.addWeighted(full_img, 1.0-mask_ratio, edge_im.astype(np.uint8),mask_ratio, 0)


    return edge_im

def draw_fullimg_mask_edge_4gpt(full_img:np.ndarray,
                                crop_box,
                                mask,
                                mask_color_bgr = (0, 0, 255),
                                mask_ratio : float = 0.1,
                                edge_color_bgr = (255, 0, 0)):
    y_size, x_size, channel = full_img.shape
    assert channel == 3
    assert 0.0 <= mask_ratio <= 1.0
    x, y, w, h = crop_box
    # mask is y 1st, x 2nd
    full_mask = np.zeros( [y_size,x_size],dtype=np.uint8 )
    blank_img = np.zeros_like(full_img)

    full_mask[y:y + h, x:x + w] = mask

    masked_img = np.where(full_mask[...,None], np.asarray((255,255,255)), blank_img)
    masked_img_bi = cv.cvtColor(masked_img.astype(np.float32), cv.COLOR_BGR2GRAY)
    edge_im = cv.Canny(masked_img_bi.astype(np.uint8),50,300)

    color_mask = np.where(full_mask[...,None], np.asarray(mask_color_bgr), blank_img).astype(np.uint8)

    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    new_img = cv.addWeighted(full_img, 1.0 - mask_ratio, color_mask, mask_ratio, 0)
    new_img[edge_im != 0] = np.asarray(edge_color_bgr)

    now = datetime.datetime.now()
    dt = now.strftime('%m%d-%H%M-%S-%f')[:-3]
    cv.imwrite(f"image_4_gpt4v/{dt}.jpg", new_img)
    print(f"file created{dt}")

    return new_img


class intereaction_GUI:
    def __init__(self, image_path, annotation_path):
        self.image_file_name = image_path
        self.raw_image = cv.imread(image_path)
        self.y, self.x, c = self.raw_image.shape
        self.annots = load_annotation(annotation_path)
        self.cursor = (self.x // 2, self.y // 2)

        pygame.init()

        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        pygame.display.set_caption("Joystick example")

        self.joystick = pygame.joystick.Joystick(0)  # create a joystick instance
        self.joystick.init()  # init instance

        self.touch_playback_music = open(touch_playback_sound_path, "rb")

        # Used to manage how fast the screen updates.
        self.clock = pygame.time.Clock()
        self.annots_current = None
        self.focus_annot_index = None
        self.mode = MODE_FULLVIEW

    def process_pyg_event_readout(self):
        now = datetime.datetime.now()
        dt = now.strftime('%H:%M:%S.%f')[:-3]

        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.JOYBUTTONDOWN:
                print("\nJoystick button pressed.")
                if event.button == 3:
                    print("Button Y - cursor pick")
                    self.annots_current = self.read_cursor()
                    if len(self.annots_current) == 0:
                        quick_speak(L.get_str("cursor_pick_empty"))
                        self.mode = MODE_FULLVIEW
                        self.focus_annot_index = None
                        pygame.event.clear()
                        make_end_sound()
                        break
                    # enter sep
                    quick_speak(L.get_str("enter_separate_mode"))
                    self.mode = MODE_SEPARATE
                    self.focus_annot_index = 0
                    pygame.event.clear()
                    make_end_sound()
                    break
                elif event.button == 5:
                    quick_speak(L.get_str("now_readout_mode"))

            elif event.type == pygL.KEYDOWN and event.key == pygL.K_ESCAPE:
                print("\nEsc pressed.")
                quit()
            elif event.type == pygL.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                self.move_cursor2(x, y)
                # print(f"mouse : {x},{y}")

    def process_pyg_event_separate(self):
        now = datetime.datetime.now()
        dt = now.strftime('%H:%M:%S.%f')[:-3]

        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.JOYBUTTONDOWN:
                print("\nJoystick button pressed.")
                if event.button == 3:  # <Y>
                    print("Button Y - back to readout")
                    quick_speak(L.get_str("return_2_readout_mode"))
                    self.annots_current = None
                    self.focus_annot_index = None
                    self.mode = MODE_FULLVIEW
                    pygame.event.clear()
                    make_end_sound()
                    break
                elif event.button == 2:  # <X>
                    print("Button X - enter touch mode")
                    quick_speak(L.get_str("enter_touch_mode"))
                    self.touch_mode_resource_init()
                    pygame.event.clear()
                    make_end_sound()
                    break
                elif event.button == 1:  # <B>
                    if not self.annots_current:
                        quick_speak(L.get_str("no_object_selected"))
                        break
                    else:
                        self.focus_annot_index =  (self.focus_annot_index + 1) % len(self.annots_current)

                        composited_descr_str = get_list_description_GPT(focus_annot_index=self.focus_annot_index,
                                                                        len_annots_current=len(self.annots_current),
                                                                        class_name=
                                                                   self.annots_current[self.focus_annot_index][
                                                                       'class_name'],
                                                                        class_proposals=
                                                                   self.annots_current[self.focus_annot_index][
                                                                       'class_proposals'])

                        print(f"STR:{composited_descr_str}")
                        quick_speak(composited_descr_str)
                        make_end_sound()
                    break
                elif event.button == 0:  # <A>
                    if not self.annots_current:
                        quick_speak(L.get_str("no_object_selected"))
                        make_end_sound()
                        break
                    else:
                        self.focus_annot_index = (self.focus_annot_index + 1) % (len(self.annots_current))
                        quick_speak(L.get_str("please_wait"))
                        annot = self.annots_current[self.focus_annot_index]
                        img_with_mask_edge = draw_fullimg_mask_edge_4gpt(self.raw_image,
                                                                         annot['crop_box'],
                                                                         annot['segmentation'],
                                                                         )
                        composited_descr_str =  get_GPT4v_description_obj(img_with_mask_edge,supportedLocals_fullname[LANG.value])

                        quick_speak(openup_description_GPT4V(composited_descr_str))

                        make_end_sound()
                    break


            elif event.type == pygL.KEYDOWN and event.key == pygL.K_ESCAPE:
                print("\nEsc pressed.")
                quit()
            elif event.type == pygL.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                self.move_cursor2(x, y)
                # print(f"mouse : {x},{y}")

            elif event.type == pygL.JOYAXISMOTION:
                axis_0 = self.joystick.get_axis(2)
                axis_1 = self.joystick.get_axis(3)
                axis_0 = sgn(axis_0)
                axis_1 = sgn(axis_1)
                # axis_0 = axis_0 if abs(axis_0) >= JOYSTICK_DEADZONE else 0
                # axis_1 = axis_1 if abs(axis_1) >= JOYSTICK_DEADZONE else 0

                print(f"{dt} hats 0/1 : {axis_0:.2f} {axis_1:.2f}")

                break

    def process_pyg_event_touch(self):
        now = datetime.datetime.now()
        dt = now.strftime('%H:%M:%S.%f')[:-3]

        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.JOYBUTTONDOWN:
                # print("\nJoystick button pressed.")
                if event.button == 2:
                    print("Button X - exit touch mode")
                    self.exit_touch()
                    quick_speak(L.get_str("enter_separate_mode"))
                    make_end_sound()
                    break

            elif event.type == pygL.KEYDOWN and event.key == pygL.K_ESCAPE:
                print("\nEsc pressed.")
                quit()
            elif event.type == pygL.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                self.move_cursor2(x, y)
                flag = is_on_mask(self.cursor, self.annots_current[self.focus_annot_index]['crop_box'],
                                  self.annots_current[self.focus_annot_index]['segmentation'])
                if flag is None:
                    out_of_frame_sound.play()
                    return
                if flag:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
                else:
                    pygame.mixer.music.stop()  # pass

    def read_input(self):
        pass

    def get_corresponding_masks(self, x, y):
        annot_list = []
        for annot in self.annots:
            if not is_in_bbox((x, y), annot['bbox']):
                continue
            if not is_on_mask((x, y), annot['crop_box'], annot['segmentation']):
                continue
            annot_list.append(annot)
        annot_list.sort(key=lambda item: item['area'])
        return annot_list

    def respond_to_pos(self, x, y):

        cr_masks = self.get_corresponding_masks(x, y)
        if cr_masks:
            print(f"pos : {x} , {y}")
        obj_count_str = get_obj_count_str(
                                          len_annots_current=len(cr_masks),
                                          )
        print(f"STR:{obj_count_str}")

        quick_speak(obj_count_str)
        # quick_speak( f"number {i+1} of {len(cr_masks)} : type {annot['class_name']} , description: " + ','.join( annot['class_proposals'] ) )
        make_partial_end_sound()
        """
        for i, annot in enumerate(cr_masks):
            obj_count_str = get_obj_count_str(focus_annot_index=i,
                                                            len_annots_current=len(cr_masks),
                                                            class_name=annot['class_name'],
                                                            class_proposals=annot['class_proposals'])
            print(f"STR:{obj_count_str}")

            quick_speak(obj_count_str)
            # quick_speak( f"number {i+1} of {len(cr_masks)} : type {annot['class_name']} , description: " + ','.join( annot['class_proposals'] ) )
            make_partial_end_sound()
        """
        return cr_masks

    def get_drawn_img(self):
        image = self.raw_image.copy()
        draw_cursor(image, self.cursor[0], self.cursor[1])
        return image

    def get_masked_drawn_img(self, annot):
        image = self.raw_image.copy()
        # print(annot)
        image, full_mask = draw_fullimg_mask(image, annot['crop_box'], annot['segmentation'], )
        draw_cursor(image, self.cursor[0], self.cursor[1])
        return image

    def exit_touch(self):
        assert self.mode == MODE_TOUCH
        self.mode = MODE_SEPARATE
        # pygame.mixer.music.unload()

    def touch_mode_resource_init(self):
        assert not self.mode == MODE_TOUCH
        self.mode = MODE_TOUCH
        pygame.mixer.music.load(touch_playback_sound_path)

    def get_edgeed_drawn_img(self, annot):
        image = self.raw_image.copy()
        #print(annot)
        bi_image = draw_fullimg_mask_edge(image, annot['crop_box'], annot['segmentation'],)
        bi_image = cv.cvtColor(bi_image, cv.COLOR_GRAY2BGR)
        return bi_image

    def show(self):
        done = False
        img = self.get_drawn_img()
        self.screen.blit(convert_opencv_img_to_pygame(img),
                         (0, 0))
        pygame.display.update()

        while not done:
            if self.mode == MODE_FULLVIEW:

                self.process_pyg_event_readout()
            elif self.mode == MODE_TOUCH:
                self.process_pyg_event_touch()
            elif self.mode == MODE_SEPARATE:

                self.process_pyg_event_separate()
            else:
                pass
            self.screen.fill((0, 0, 0))
            """
            if not self.annots_current:
                new_img = self.get_drawn_img()
            else:
                new_img = self.get_masked_drawn_img(self.annots_current[self.focus_annot_index])
            """
            if self.mode == MODE_TOUCH:
                #new_img = self.get_bi_masked_drawn_img(self.annots_current[self.focus_annot_index])
                new_img = self.get_edgeed_drawn_img(self.annots_current[self.focus_annot_index])
            elif not self.annots_current:
                new_img = self.get_drawn_img()
            else:
                new_img = self.get_masked_drawn_img(self.annots_current[self.focus_annot_index])

            self.screen.blit(convert_opencv_img_to_pygame(new_img),
                             (0, 0))
            # Go ahead and update the screen with what we've drawn.
            pygame.display.update()

            # Limit to 30 frames per second.
            self.clock.tick(5)
            pygame.time.wait(5)

    def mouse_click(self, event, x, y,
                    flags, param):
        # to check if left mouse
        # button was clicked
        if event == cv.EVENT_LBUTTONDOWN:
            # font for left click event
            font = cv.FONT_HERSHEY_TRIPLEX
            LB = 'Left Button'
            self.respond_to_pos(x, y)
            # display that left button
            # was clicked.

    def read_cursor(self):
        return self.respond_to_pos(self.cursor[0], self.cursor[1])

    def move_cursor(self, dx, dy):
        x = self.cursor[0]
        y = self.cursor[1]
        self.cursor = (x + dx, y + dy)

    def move_cursor2(self, x, y):

        self.cursor = (x, y)

    def __del__(self):
        pygame.quit()
        self.touch_playback_music.close()




if __name__ == '__main__':
    gui = intereaction_GUI(img_path, json_path)
    gui.show()
