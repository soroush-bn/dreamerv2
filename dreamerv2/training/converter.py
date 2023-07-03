import numpy as np


def three_channel_converter(obs):
    channeled = np.zeros((3, 10, 10))
    compact = convert_to_compact(obs)
    compact = compact / 255
    channeled[0, :, 0] = compact[:, 1]
    channeled[1, :, 9] = compact[:, 8]
    channeled[2] = compact
    channeled[2, :, 1] = 0
    channeled[2, :, 8] = 0
    return channeled


def convert_to_compact(frame):
    p_obs = preprocess_single(frame)
    bw_obs = make_bw_frame(p_obs)
    converted_obs = conv2dpong(bw_obs)
    return converted_obs


def make_bw_frame(p_obs):
    p_obs = p_obs.astype(int)
    ball_index = np.where(p_obs == 158)
    pads_index_right = np.where(p_obs == 61)
    pads_index_left = np.where(p_obs == 45)
    bw_obs = np.zeros(p_obs.shape)
    bw_obs[ball_index] = 255
    bw_obs[pads_index_right] = 255
    bw_obs[pads_index_left] = 255
    return bw_obs


def conv2dpong(input):
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


def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    # print('image[34:-16:2,::2].shape: ', image[34:-16:2,::2].shape)
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1)
    return img
