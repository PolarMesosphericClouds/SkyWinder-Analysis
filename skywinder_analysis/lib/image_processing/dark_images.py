import skywinder_analysis.lib.tools.blosc_file as blosc_file

def get_dark_image(camera_number, exposure):
    if camera_number == 99:
        camera_number = 7
    dark_100ms, _ = blosc_file.load_blosc_image('/home/christopher/SkyWinder-Analysis/skywinder_analysis/resources/dark_images/c{0}_dark_image_100ms'.format(camera_number))
    dark_1s, _ = blosc_file.load_blosc_image('/home/christopher/SkyWinder-Analysis/skywinder_analysis/resources/dark_images/c{0}_dark_image_1s'.format(camera_number))    
    dark = dark_100ms.astype('float') + float((exposure - 100000)/900000)*(dark_1s.astype('float') - dark_100ms.astype('float'))
    return dark
