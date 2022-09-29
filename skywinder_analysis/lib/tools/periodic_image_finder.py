import glob
import os
import skywinder_analysis


def find_periodic_images(camera_number, start_timestamp, stop_timestamp, interval):
    all_filenames = get_all_image_filenames(camera_number)
    fn_list = []
    current_timestamp = start_timestamp
    for fn in all_filenames:
        try:
            if current_timestamp <= int(fn.split('=')[-1][:10]):
                fn_list.append(fn.strip('\n'))
                current_timestamp += interval
                if current_timestamp >= stop_timestamp:
                    break
        except Exception as e:
            print(fn)
            raise(e)
    return fn_list


def find_periodic_images_full_fn(camera_number, start_timestamp, stop_timestamp, interval):
    all_filenames = get_all_image_filenames(camera_number)
    fn_list = []
    current_timestamp = start_timestamp
    for fn in all_filenames:
        if len(fn.split('=')) > 1:
            if current_timestamp <= int(fn.split('=')[-1][:10]):
                if int(fn.split('=')[-1][:10]) > stop_timestamp:
                    break
                fn_list.append(fn.strip('\n'))
                current_timestamp += interval
                if current_timestamp >= stop_timestamp:
                    break
    full_fns = []
    if camera_number < 4:
        direc = '/data/mounted_filesystems/nas1/c{0}'.format(camera_number)
    elif camera_number == 99:
        direc = '/data/mounted_filesystems/nas1/piggyback'
    else:
        direc = '/data/mounted_filesystems/nas2/c{0}'.format(camera_number)
    for fn in fn_list:
        full_fns.append(os.path.join(direc, fn))

    return full_fns

def find_all_images(camera_number, start_timestamp, stop_timestamp):
    all_filenames = get_all_image_filenames(camera_number)
    fn_list = []
    for fn in all_filenames:
        if start_timestamp <= int(fn.split('=')[-1][:10]):            
            if int(fn.split('=')[-1][:10]) >= stop_timestamp:
                break
            fn_list.append(fn.strip('\n'))
    return fn_list


INDIVIDUAL_CAMERA_DIRNAMES = {
    '1': '2018-07-08_035140',
    '2': '2018-07-07_100519',
    '3': '2018-07-07_100509',
    '4': '2018-07-07_100519',
    '6': '2018-07-07_100513',
    '7': '2018-07-08_035010'}


#def get_fn_end(fn):
#    return fn.split('/')[-1]

def get_fn_end(fn):
    return fn.split('=')[-1]


def get_all_image_filenames(camera_number):
    all_filenames = []
    if camera_number == 5:
        # Camera 5 was restarted and thus has two sets of filenames and directories
        for i in range(1, 6):
            for date in ['2018-07-07_100515', '2018-07-13_031421']:
                with open(os.path.join(skywinder_analysis.__path__[0], 'resources', 'new_list_of_all_filenames', 'c5',
                                       ('c5_data%d_%s.txt' % (i, date)))) as f:
                    fns = f.readlines()
                    for fn in fns:
                        all_filenames.append(
                            os.path.join(('data%d' % i), date, fn))
    elif camera_number == 99:
        # Piggyback is "camera 99"
        # Piggyback was restarted and thus has two sets of filenames and directories
        for i in range(1, 6):
            for date in ['2019-12-15_144434', '2019-12-19_012658']:
                with open(os.path.join(skywinder_analysis.__path__[0], 'resources', 'list_of_all_filenames', 'c99',
                                       ('c99_data%d_%s.txt' % (i, date)))) as f:
                    fns = f.readlines()
                    for fn in fns:
                        all_filenames.append(
                            os.path.join(('data%d' % i), date, fn))
    else:
        # Cameras 1, 2, 3, 4, 6, 7
        for i in range(1, 6):
            with open(os.path.join(skywinder_analysis.__path__[0], 'resources', 'new_list_of_all_filenames',
                                   ('c%d' % camera_number),
                                   ('c%d_data%d.txt' % (camera_number, i)))) as f:
                fns = f.readlines()
                for fn in fns:
                    all_filenames.append(
                        os.path.join(('data%d' % i), INDIVIDUAL_CAMERA_DIRNAMES[str(camera_number)], fn))
    all_filenames.sort(key=get_fn_end)
    return all_filenames
