# Yueqing Xuan
# 1075355

# This file receives the visual information from the visual pipeline, and format
# the visual information so that it can be used for entity linking

import socket
import sys


# process the visual info received from the visual pipeline
def split_visual_info(point_cloud):

    assert((len(point_cloud)-1) % 4 == 0)

    num_of_items = int((len(point_cloud)-1) / 4)
    entity_list = []

    for i in range(num_of_items):

        entity_name = point_cloud[i*4]
        x = float(point_cloud[i*4+1][25:-2])
        y = float(point_cloud[i*4+2][5:-2])
        z = float(point_cloud[i*4+3][5:-2])

        position = (x,y,z)
        entity = {
            "name": entity_name,
            "position": position
        }

        entity_list.append(entity)

    return entity_list


# get info from visual pipeline
def receive_from_zed():

    laptop_ip = socket.gethostname()
    port = 10000

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((laptop_ip, port))
    s.settimeout(0.00001)

    try:
        data, addr = s.recvfrom(4096)
        detection_data = str(data)
        z = detection_data.split('//')
        detection = z[0]
        point_cloud = z[1]
        point_cloud_split = point_cloud.split(",")

        entities_detected = split_visual_info(point_cloud_split)
    except socket.timeout as e:
        return None
    finally:
        s.close()
    return entities_detected


# print the visual information
if __name__ == '__main__':
    while True:
        try:
            txt = input("Type y to continue.")
            if txt == "y":
                receive_from_zed()
        except KeyboardInterrupt:
            print("Exit!")
            sys.exit(0)
