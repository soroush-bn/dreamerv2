import numpy as np


# todo fix this by importing

class Converter():
    def __init__(self, pre_right, pre_left, pre_ball):
        self.pre_right = pre_right
        self.pre_left = pre_left
        self.pre_ball = pre_ball

    def three_channel_converter(self, obs):
        channeled = np.zeros((3, 10, 10))
        compact = self.convert_to_compact(obs)
        compact = compact / 255
        channeled[0, :, 0] = compact[:, 1]
        channeled[1, :, 9] = compact[:, 8]
        channeled[2] = compact
        channeled[2, :, 1] = 0
        channeled[2, :, 8] = 0
        return channeled

    def convert_to_compact(self, frame):
        p_obs = self.preprocess_single(frame)
        bw_obs = self.make_bw_frame(p_obs)
        xpr, ypr = self.find_paddle_right(bw_obs, self.pre_right)

        # print("right :" + str(xpr) + "," + str(ypr))
        xpl, ypl = self.find_paddle_left(bw_obs, self.pre_left)
        # print("left :" + str(xpl) + "," + str(ypl))
        xb, yb = self.find_ball(bw_obs, self.pre_ball)
        map = np.zeros((10, 10))
        if xpr == 10:
            xpr -= 1
        if xpl == 10:
            xpl -= 1
        if xpr != None and ypr != None:
            map[xpr, ypr] = 1
            if (xpr == 0):
                map[xpr + 1, ypr] = 1
                map[xpr + 2, ypr] = 1
            elif (xpr == 9):
                map[xpr - 1, ypr] = 1
                map[xpr - 2, ypr] = 1
            else:
                map[xpr - 1, ypr] = 1
                map[xpr + 1, ypr] = 1

        if xpl != None and ypl != None:
            map[xpl, ypl] = 1
            if (xpl == 0):
                map[xpl + 1, ypl] = 1
                map[xpl + 2, ypl] = 1
            elif (xpl == 9):
                map[xpl - 1, ypl] = 1
                map[xpl - 2, ypl] = 1
            else:
                map[xpl - 1, ypl] = 1
                map[xpl + 1, ypl] = 1
        if xb != None and yb != None:
            map[xb, yb] = 1
        map = np.array(map, dtype=bool)
        self.pre_right = (xpr, ypr)
        self.pre_left = (xpl, ypl)
        self.pre_ball = (xb, yb)
        return np.expand_dims(map, axis=0)

    def find_ball(self, bw_obs, previous_position: tuple):
        bw_obs = bw_obs[:, 10:70]
        mask = np.ones((8, 6))
        for i in range(10):
            for j in range(10):
                res = bw_obs[i * 8:(8 + i * 8), j * 6:(6 + j * 6)] * mask
                if res.sum() == 2 * 255:
                    if j != previous_position[1]:
                        return int(i), int(j)
        return previous_position

    def find_paddle_right(self, bw_obs, previous_position: tuple):
        # plt.imshow(bw_obs)
        bw_obs2 = bw_obs[:, 70:72]
        mask = np.ones((8, 2))
        for i in range(bw_obs2.shape[0] - 8):
            res = bw_obs2[i:(8 + i), :] * mask
            if res.sum() == 16 * 255:
                bw_obs[i:(8 + i), 70:72] = 0
                return int(i / 8), 9
        return previous_position

    def find_paddle_left(self, bw_obs, previous_position: tuple):
        bw_obs2 = bw_obs[:, 8:10]

        mask = np.ones((8, 2))
        for i in range(bw_obs2.shape[0] - 8):
            res = bw_obs2[i:(8 + i), :] * mask
            if res.sum() == 16 * 255:
                bw_obs[i:(8 + i), 9:11] = 0
                return int(i / 8), 0
        return previous_position

    def preprocess_single(self, image, bkg_color=np.array([144, 72, 17])):
        # print('image[34:-16:2,::2].shape: ', image[34:-16:2,::2].shape)
        img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1)
        return img

    def make_bw_frame(self, p_obs):
        p_obs = p_obs.astype(int)
        ball_index = np.where(p_obs == 158)
        pads_index_right = np.where(p_obs == 61)
        pads_index_left = np.where(p_obs == 45)
        bw_obs = np.zeros(p_obs.shape)
        bw_obs[ball_index] = 255
        bw_obs[pads_index_right] = 255
        bw_obs[pads_index_left] = 255
        return bw_obs

    def conv2dpong(self, input):
        conv = np.zeros((10, 10))
        i = 0
        j = 0
        # for c in range(100):
        for i in range(10):
            for j in range(10):
                conv[i, j] = input[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].sum()

        conv[np.where(conv > 0)] = 255
        return conv

# plt.imshow(conv)


#
# def convert_to_compact(frame):
#     result = np.zeros((10, 10))
#     p_obs = preprocess_single(frame)
#     bw_obs = make_bw_frame(p_obs)
#     converted_obs = conv2dpong(bw_obs)
#     result[:, 0] = converted_obs[:, 1]
#
#     result[:, 9] = converted_obs[:, 8]
#     for i in range(1,9):
#         c=True
#         for j in result[:,i]:
#             if j!= 0:
#                 if c:
#                     c=False
#                 else:
#                     j=0
#
#     return result/255
#
