import h5py as hdf
import numpy as np
from scipy import ndimage
import os
import RBJudge
import sys

def import_data(I, S):
    Initial = hdf.File(I, "r")
    FeatureIds = Initial.get("/DataContainers/SyntheticVolume/CellData/FeatureIds")[()]
    Quats = Initial.get("/DataContainers/SyntheticVolume/CellFeatureData/AvgQuats")[()]
    Stats = hdf.File(S, 'r')
    Grainsize = Stats.get('grainsize')[()]
    return FeatureIds, Grainsize, Quats

def Quants_type(Quats):
    quat = np.count_nonzero(Quats, axis=1) - 1
    quat_l = list(enumerate(quat, start=0))
    del(quat_l[0])
    numtype_d = dict((x, y) for x, y in quat_l)
    return numtype_d

def get_grain_size(system, grain, **step):
    true_step = 0 if not step else int(list(step.values())[0]*2.5)
    return system.Grainsize[true_step, grain]

def get_grain_size_sum(system, *grain):
    grain_size = []
    for i in grain:
        its_size = get_grain_size(system, i)
        grain_size.append(its_size)
    return np.sum(grain_size)

def get_nnd_key(dict, value):
    return [k for k, v in dict.items() if value in v]

def whether_it_disappear(ID, system):  #delete when predict
    bottom_line = system.Grainsize[250, ID]
    grain_line = system.Grainsize[:, ID]
    if bottom_line == 0:
        return np.argwhere(grain_line == 0)[0][0]
    else:
        return 250

def margin(system, *grain_id):
    if len(grain_id) == 1:
        grain_matrix = np.argwhere(system.FeatureIds == grain_id)
        copy_F = system.FeatureIds.copy()
        for row, column in grain_matrix:
            copy_F[row][column] = 0
        distance = ndimage.distance_transform_edt(copy_F)
        return np.argwhere(distance == 1)
    else:
        grains_matrix = np.argwhere(system.FeatureIds == grain_id[0])
        for num in range(1, len(grain_id)):
            next_grain = grain_id[num]
            grains_matrix = np.concatenate((grains_matrix, np.argwhere(system.FeatureIds == next_grain)), axis=0)
        copy_F = system.FeatureIds.copy()
        for row, column in grains_matrix:
            copy_F[row][column] = 0
        distance = ndimage.distance_transform_edt(copy_F)
        return np.argwhere(distance == 1)

def nearest_neighbor(system, grain_margin):
    number_list = []
    for row, column in grain_margin:
        number = system.FeatureIds[row][column]
        number_list.append(number)
    nearest_neighbor = np.unique(np.array(number_list))
    return nearest_neighbor

def find_position_ID(system, position):
    row = position[0]
    column = position[1]
    return system.FeatureIds[row][column]

def find_new_neighbor(system, nn_dictionary):
    neighbor_id_list = sum(list(nn_dictionary.values()), [])
    zero_matrix = []
    for grain in neighbor_id_list:
        grain_matrix = np.argwhere(system.FeatureIds == grain)
        zero_matrix.extend(grain_matrix)
    zero_array = np.array(zero_matrix)
    copy_F = system.FeatureIds.copy()
    for row, column in zero_array:
        copy_F[row][column] = 0
    distance = ndimage.distance_transform_edt(copy_F)
    position_all = np.argwhere(distance == 1)   # yin out_margin
    id_list = []
    for position in position_all:
        ID = find_position_ID(system, position)
        id_list.append(ID)
    new_neighbor_id_list = list(np.unique(id_list))
    return new_neighbor_id_list

def find_nn_d(system, grain_id, max_nn):
    grain_id = [grain_id] if isinstance(grain_id, list) == False else grain_id
    nn_dictionary = {}
    nn_dictionary[0] = grain_id
    grain_margin = margin(system, grain_id)
    first_nn = nearest_neighbor(system, grain_margin)
    nn_dictionary[1] = list(first_nn)
    number = 2
    while number != max_nn + 1:
        this_nn = find_new_neighbor(system, nn_dictionary) # ,in_out_count_d
        nn_dictionary[number] = list(this_nn)
        number += 1
    return nn_dictionary

def find_cross(system, position):
    row_up = 0
    row_down = np.shape(system.FeatureIds)[0] - 1
    col_left = 0
    col_right = np.shape(system.FeatureIds)[1] - 1
    row = position[0]
    col = position[1]
    cross = {}  ## 0=up, 1=down, 2=left, 3=right
    if row == row_up:
        cross[0] = [row_down, col]
    else:
        cross[0] = [row - 1, col]
    if row == row_down:
        cross[1] = [row_up, col]
    else:
        cross[1] = [row + 1, col]
    if col == col_left:
        cross[2] = [row, col_right]
    else:
        cross[2] = [row, col - 1]
    if col == col_right:
        cross[3] = [row, col_left]
    else:
        cross[3] = [row, col + 1]
    return cross

def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])

def neighbor_grain_size(system, neighbor_id_list):
    first_row = system.Grainsize[0, :]
    size_d = {}
    for id in neighbor_id_list:
        size_d[id] = first_row[id]
    return size_d

def meta_neighbor_grain_size(system, neighbor_id_list):
    size_d = {}
    for id in neighbor_id_list:
        its_size = np.argwhere(system.FeatureIds == id).shape[0]
        size_d[id] = its_size
    return size_d

def neighbor_grain_type(system, neighbor_id_list):
    id_type_d = {}
    for grain in neighbor_id_list:
        id_type_d[grain] = system.numtype_d[grain]
    return id_type_d

def neighbor_grain_length(system, grain_id, neighbor_id_list):
    out_margin = margin(system, grain_id)
    grain_matrix = np.argwhere(system.FeatureIds == grain_id)
    grain_list = grain_matrix.tolist()
    length_d = dict.fromkeys(neighbor_id_list, 0)
    for position in out_margin:
        nn_id = find_position_ID(system, position)
        cross = find_cross(system, position)
        in_grain = [k for k,v in cross.items() if v in grain_list]
        length_d[nn_id] += len(in_grain)
    return length_d

def get_NRF(grain):
    n = [k for k, v in grain.nn_type.items() if v == 2]
    return len(n) / len(grain.nn)

def get_SRF(system):
    red_grain = [k for k,v in system.numtype_d.items() if v == 2]
    counts = system.Quats.shape[0] - 1
    return len(red_grain) / counts

def get_LF(system, grain_id, other_grain_id, **step): # p1 for grain considered to be disappear, p2 for grain push(candidate)
    has_step = list(step.values())[0]
    grain = Grain(system, grain_id) if not step else Step_Grain(system, grain_id, has_step)
    other_grain = Grain(system, other_grain_id) if not step else Step_Grain(system, other_grain_id, has_step)
    share_nn = [nn for nn in grain.nn if nn in other_grain.nn]
    margin_sum = sum(list(grain.nn_length.values()))
    count_board = np.zeros((3, 3))
    share_nn.insert(0, other_grain.ID)
    for row in range(len(share_nn)):
        this_id = share_nn[row]
        count_board[row, 0] = grain.nn_length[this_id] / margin_sum
        count_board[row, 1] = grain.nn_type[this_id]
        count_board[row, 2] = this_id
    return count_board

def update_INN(self, candidate):
    pre_nn = self.nn
    white_grain_list = candidate.contain
    whited_nn = [nn for nn in pre_nn if nn in white_grain_list]
    num_whited_nn = len(whited_nn)
    return self.INN - num_whited_nn

def get_opposite_nodes(system, grain_id, other_grain_id, **step):
    has_step = list(step.values())[0]
    grain = Grain(system, grain_id) if not step else Step_Grain(system, grain_id, has_step)
    other_grain = Grain(system, other_grain_id) if not step else Step_Grain(system, other_grain_id, has_step)
    share_nn = [nn for nn in grain.nn if nn in other_grain.nn]
    return share_nn

def get_two_largest(self, id_list):
    if len(id_list) != 2:
        IRS_d = {}
        for i in id_list:
            its_IRS = self.nn_size[i]
            IRS_d[i] = its_IRS
        largest = max(IRS_d, key=IRS_d.get)
        id_list.remove(largest)
        second_largest = max(IRS_d, key=IRS_d.get)
        return [largest, second_largest]
    else:
        return id_list



# def out_count_board(count_board):
#     return print(count_board)

class System(object):
    def __init__(self, CATA, SUBCATA, SAMPLE):  #CATA = scratch, SUBCATA = AGGmodel-299697-
        self.filepath = '/Volumes/Samsung_T5/hippolyta/' + str(CATA) + '/' + str(SUBCATA) + '-' + str(int(SAMPLE))  #change the middle
        # for terminal
        self.SAMPLE = SAMPLE[4:12]
        # self.filepath = '/Volumes/Samsung_T5/hippolyta/' + str(CATA) + '/' + str(SUBCATA) + '/' + str(SAMPLE)
        # self.filepath = '/Volumes/Samsung_T5/hippolyta/' + str(CATA) + '/' + str(SUBCATA) + '/' + str(SAMPLE)

        # for pycharm
        # self.SAMPLE = SAMPLE
        # self.filepath = '/Volumes/Samsung_T5/hippolyta/' + str(CATA) + '/' + '/run_' + str(SAMPLE)[0:4] + '/run_' + str(SAMPLE)

        self.I = self.filepath + '/initial.dream3d'
        self.S = self.filepath + '/stats.h5'
        self.FeatureIds, self.Grainsize, self.Quats = import_data(self.I, self.S)
        self.numtype_d = Quants_type(self.Quats)
        self.candidate = [k for k, v in self.numtype_d.items() if v == 0][0]
        self.SRF = get_SRF(self)
        self.candidate_last_size = self.Grainsize[250, self.candidate]

    def identification(self):  ###common use -- usually candidate ID
        first_row = self.Grainsize[0, :]
        # initial_mean = np.mean(first_row) * (1 + 1 / (self.Grainsize.shape[1] - 1))
        last_row = self.Grainsize[self.Grainsize.shape[0] - 1, :]
        last_row_mean = np.mean(last_row[np.where(last_row != 0)])
        # initial_relative_size = first_row[ID] / initial_mean
        if last_row[self.candidate] > last_row_mean * 6:
            return True  # 1 = True
        elif last_row[self.candidate] < last_row_mean * 6 and last_row[self.candidate] > np.sum(first_row) * 0.8:
            return True
        else:
            return False  # 0 = False


    def initial_decision(self):
        #condition1 = NRF >=0.4, NRF<=0.1, nn>=10
        candidate = Grain(self, self.candidate)
        if candidate.NRF > 0.4:
            return True # AGG
        elif self.SRF > 0.45:
            return True # AGG
        elif candidate.NRF == 0:
            return False
        elif candidate.NRF < 0.13 and self.SRF < 0.15:
            return False
        else:
            return "check candidate"

    def showvideo(self):
        start_directory = str(self.filepath)
        os.system('open ' + start_directory + '/grains.mov')
        return

class God_System(System):
    def __init__(self, CATA, SUBCATA, SAMPLE, step):
        super(God_System, self).__init__(CATA, SUBCATA, SAMPLE)
        self.step = step
        self.filename = str(self.filepath + '/' + "agg_dump000%.3d.dream3d" % self.step) #100?250?
        self.FeatureIds = self.get_this_FeatureIds()

    def get_this_FeatureIds(self):
        this_dream3d = hdf.File(self.filename, 'r')
        this_FeatureIds = this_dream3d.get("DataContainers/SyntheticVolume/CellData/FeatureIds")[()]
        its_shape = np.shape(this_FeatureIds)
        return this_FeatureIds.reshape(its_shape[0], its_shape[1])

    def get_check_dream_list(self, grain):
        step_equal_zero = np.argwhere(self.Grainsize[:, grain] != 0)[-1, :]
        which_sub_dream_to_check = np.arange(0, round(
            int(step_equal_zero) / 2.5) + 1)  # +2 when need the next dream data ! for check low_nn only
        return ["agg_dump000%.3d.dream3d" % i for i in which_sub_dream_to_check]

    def get_grain_size(self, grain):
        return self.Grainsize[0, grain]

class Grain(object):
    def __init__(self, system, ID):
        # thing that will not be change
        self.ID = ID
        self.IRS = get_grain_size(system, self.ID)
        self.type = system.numtype_d[self.ID]
        self.margin = margin(system, self.ID)
        self.nn = nearest_neighbor(system, self.margin)
        self.nn_size = neighbor_grain_size(system, self.nn)
        self.nn_type = neighbor_grain_type(system, self.nn)
        self.nn_length = neighbor_grain_length(system, self.ID, self.nn)
        self.INN = len(self.nn)
        self.nnd = find_nn_d(system, self.ID, 3)
        self.NRF = get_NRF(self)
        self.meta_dis = self.whether_it_disappear(system)
        #

    def whether_it_disappear(self, system):  #delete when predict
        bottom_line = system.Grainsize[250, self.ID]
        grain_line = system.Grainsize[:, self.ID]
        if bottom_line == 0:
            return np.argwhere(grain_line == 0)[0][0]
        else:
            # return 250
            return False

    def get_opposite_nodes(self, system, other_grain_id):
        other_grain = Grain(system, other_grain_id)
        its_nn = self.nn
        others_nn = other_grain.nn
        share_nn = [nn for nn in its_nn if nn in others_nn]
        return share_nn

    # def structure_dis(self):

class Step_Grain(Grain):
    def __init__(self, system, ID, step):
        super().__init__(system, ID)
        # thing that will not be change
        self.step = step
        self.IRS = get_grain_size(system, self.ID, step=self.step) ##need change?

####begin
class White(Grain):
    def __init__(self, system, ID):
        super().__init__(system, ID)
        self.type = 0
        self.contain = [self.ID]
        self.nn2 = self.nnd[2]
        self.not_dis = []
        self.not_sure = []
        self.not_reach = []
        self.inner = self.check_inner_layer()
        self.inner_size = get_grain_size_sum(system, self.contain)
        self.RCP = get_nnd_key(self.nnd, system.candidate)[0]  #origin from Grain class

    def update_contain(self, system, *grains):
        old_grains = self.contain
        new_grains = old_grains + flatten(grains)
        self.margin = margin(system, *new_grains)
        return self.contain.extend(grains), self.margin

    def update_not_sure(self, grain):
        if grain not in self.not_sure:
            return self.not_sure.append(grain)
        else:
            return

    def check_not_sure(self):
        self.not_sure = list(set(self.not_sure) - (set(self.contain) | set(self.not_dis)))

    def update_not_dis(self, grain):
        self.not_dis.append(grain)

    def inner_step1(self,**layer):
        if layer['layer'] == 1:
            reds = [nn for nn in self.nn if self.nn_type[nn] == 2]
            blues = [nn for nn in self.nn if self.nn_type[nn] == 1]
            for red in reds:
                print(red, "red in layer 1")
                this_red = Red(this_system, red)
                this_red.pre_it_disappear(self)
            for blue in blues:
                print(blue, "blue")
                this_blue = Blue(this_system, blue)
                this_blue.pre_it_disappear(self)
            return self
        elif layer['layer'] == 2:
            reds = [nn for nn in self.nn2 if this_system.numtype_d[nn] == 2]
            # print(reds)
            for red in reds:
                print(red, "red in layer 2")
                this_red = Red(this_system, red)
                this_red.small_red(self)
            return self
        else:
            print("wrong layer number")

    def inner_step2(self):
        for i in self.not_sure:
            if this_system.numtype_d[i] == 2:
                print(i, "red round 2 check")
                this_red = Red(this_system, i)
                this_red.pre_it_disappear(self)
            elif this_system.numtype_d[i] == 1:
                print(i, "blue round 2 check")
                this_blue = Blue(this_system, i)
                this_blue.pre_it_disappear(self)
        self.check_not_sure()
        return self

    def last_turn(self):
        for i in self.not_sure:
            if this_system.numtype_d[i] == 1:
                this_blue = Blue(this_system, i)
                this_blue.last_turn(self)
            elif this_system.numtype_d[i] == 2:
                this_red = Red(this_system, i)
                this_red.pre_it_disappear(self)
        self.check_not_sure()
        return self

    def check_inner_layer(self):
        self.inner_step1(layer=1)
        # self.inner_step2()
        self.inner_step1(layer=2)
        self.inner_step2()
        self.last_turn()
        return self

class Red(Grain):
    def __init__(self, system, ID):
        super().__init__(system, ID)

    def small_red(self, candidate): #only NN2_red # add blue disappear time
        if self.INN <= 4:
            print("into not reach")
            return candidate.not_reach.append(self.ID)
        else:
            white_grain_list = candidate.contain
            front_nn = [nn for nn in self.nn if nn in white_grain_list]
            if len(front_nn) == 0:
                print("into not reach")
                return candidate.not_reach.append(self.ID)
            elif len(front_nn) == 1:
                nn = Grain(this_system, front_nn[0])
                if nn.type == 1 and self.IRS < nn.IRS:
                    print("into not reach")
                    return candidate.not_reach.append(self.ID)
                else:
                    return self.pre_it_disappear(candidate)
            else:
                front_l = []
                type_l = []
                for i in range(len(front_nn)):
                    this_nn = Grain(this_system, front_nn[i])
                    front_l.append(this_nn.IRS)
                    type_l.append((this_nn.type))
                front_l.append(self.IRS)
                if sum(type_l) == len(front_nn):
                    if min(front_l) == self.IRS:
                        print("into not reach")
                        return candidate.not_reach.append(self.ID)
                    else:
                        return self.pre_it_disappear(candidate)
                else:
                    return self.pre_it_disappear(candidate)

    def get_LF(self, candidate):  # p1 for grain considered to be disappear, p2 for grain push(candidate)
        white_grain_list = candidate.contain
        others_nn = nearest_neighbor(this_system, candidate.margin)
        share_nn = [nn for nn in self.nn if ((nn in others_nn) and (Grain(this_system,nn).INN > 3))]
        share_nn = get_two_largest(self, share_nn)
        margin_sum = sum(list(self.nn_length.values()))
        front_nn = [nn for nn in self.nn if nn in white_grain_list]
        count_nn = [front_nn] + share_nn
        count_board = np.zeros((3, 3))
        for row in range(0,len(count_nn)):
            this_id = count_nn[row]
            nn_length = sum(self.nn_length[i] for i in this_id) if isinstance(this_id, list) else self.nn_length[
                this_id]
            count_board[row, 0] = nn_length / margin_sum
            count_board[row, 1] = self.nn_type[this_id] if row != 0 else 0
            count_board[row, 2] = this_id if isinstance(this_id, np.int64) else 0
        return count_board

    def pre_it_disappear(self, candidate):
        count_board = self.get_LF(candidate)
        if self.ID in candidate.not_reach:
            return print('not reach')
        else:
            updated_INN = update_INN(self, candidate)
            if np.sum(count_board[1:2, 1] == 2) == 2: #TYPE1
                print("into contain")
                return candidate.update_contain(this_system, self.ID)
            elif np.sum(count_board[1:2, 1] == 2) == 1 and updated_INN <= 6: #TYPE2_sub1, original 7
                print("into contain")
                return candidate.update_contain(this_system, self.ID)
            else:
                # LF0 relationship
                share_boundaries = count_board[[1, 2], 0].tolist()
                # global LF0
                LF0 = count_board[0, 0].tolist()
                LFS = min(share_boundaries)
                LFL = max(share_boundaries)
                if np.sum(count_board[1:2, 1] == 2) == 0:  # TYPE3
                    # LF0, IRS, LFS, LFL, INN
                    if RBJudge.TYPE3_check(LF0, self.IRS, LFS, LFL, updated_INN) == 0:
                        print("into contain")
                        return candidate.update_contain(this_system, self.ID)
                    else:
                        print("into not sure")
                        return candidate.update_not_sure(self.ID)
                # elif np.sum(count_board[1:2, 1] == 2) == 0 and updated_INN >= 7: #TYPE3_sub2
                #     # 'LF0', 'IRS', 'LFS', 'LFL'
                #     if RBJudge.TYPE3_sub2_check(LF0, self.IRS, LFS, LFL) == 0:
                #         print("into contain")
                #         return candidate.update_contain(this_system, self.ID)
                #     else:
                #         print("into not sure")
                #         return candidate.update_not_sure(self.ID)
                print("into not sure")
                return candidate.update_not_sure(self.ID)


class Blue(Grain):
    def __init__(self, system, ID):
        super().__init__(system, ID)
        self.RCP = get_nnd_key(self.nnd, this_system.candidate)

    def last_turn(self, candidate):
        small_nn = max(self.nn_size, key=self.nn_size.get)
        print(small_nn, "in last Blue turn")
        this_small_nn = Grain(this_system, small_nn)
        # BLUE_567_check(NRF, IRS, INN, irs1, rcp1, typ1)
        if RBJudge.BLUE_567_check(self.NRF, self.IRS, self.INN, this_small_nn.IRS, this_small_nn.RCP,
                                    this_small_nn.type) == 0:
            print(this_small_nn.ID, "in last Blue turn into contain")
            return candidate.update_contain(this_system, self.ID)
        else:
            print(this_small_nn.ID, "in last Blue turn into not disappear")
            return candidate.update_not_dis(self.ID)


    def pre_it_disappear(self, candidate):
        updated_INN = update_INN(self, candidate)
        if updated_INN <= 4 and self.IRS <= 700:
            print('into contain')
            return candidate.update_contain(this_system, self.ID)
        elif self.IRS <= 400:
            print('into contain')
            return candidate.update_contain(this_system, self.ID)
        elif self.IRS >= 800:
            print('into not disappear')
            return candidate.update_not_dis(self.ID)
        elif updated_INN >= 8:
            print('into not disappear')
            return candidate.update_not_dis(self.ID)
        else:
            print('into not sure')
            return candidate.update_not_sure(self.ID)
        # else:
        #     small_nn = max(self.nn_size, key=self.nn_size.get)
        #     this_small_nn = Grain(this_system, small_nn)
        #     # BLUE_567_check(NRF, IRS, INN, irs1, rcp1, typ1)
        #     if BlueJudge.BLUE_567_check(self.NRF, self.IRS, self.INN, this_small_nn.IRS, this_small_nn.RCP,
        #                                 this_small_nn.type) == 0:
        #         print('into contain')
        #         return candidate.update_contain(this_system, self.ID)
        #     else:
        #         print('into not sure')
        #         return candidate.update_not_sure(self.ID)
####stop

def write_txt(content, filename):
    content = flatten(content)
    for line in content:
        filename.write(str(line) + ',')
    filename.write('\n')
    return

if __name__ == "__main__":
    CATA = sys.argv[1]
    SUBCATA = sys.argv[2]
    SAMPLE = sys.argv[3]
    


    # basic_info = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_info.txt', 'a')
    # basic_contain = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_contain.txt', 'a')
    # basic_notsure = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_notsure.txt', 'a')
    # basic_play = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_play.txt', 'a')
    # basic_last_size = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_last_size.txt', 'a')
    # basic_check_1106 = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_check_1106.txt', 'a')
    # basic_check_1206 = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_check_1206.txt', 'a')
    # basic_check_1006 = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_check_1006.txt', 'a')
    # basic_check_906 = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_check_906.txt', 'a')
    basic_tf_2000 = open('/Users/lvmeizhong/Desktop/hippolyta-pro/data/basic_tf_2000.txt', 'a')

    this_system = System(CATA, SUBCATA, SAMPLE)
    AGG = this_system.identification()
#     SRF = this_system.SRF
    C = Grain(this_system, this_system.candidate)
#     NRF = C.NRF
    INN = C.INN
    IRS = C.IRS
    TYP = C.type
    
#     check_content = [AGG, NRF, SRF]

#     basic_last_size_content = [this_system.SAMPLE, this_system.identification(), this_system.SRF, C.NRF, C.INN, C.IRS, C.whether_it_disappear(this_system)]






















