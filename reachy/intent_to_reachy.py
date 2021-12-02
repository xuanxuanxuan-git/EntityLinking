# Yueqing Xuan
# 1075355

# Map intents from the intent classifier to a sequence of actions to be executed by reachy.
# you can look at the note (RA week 3 -> intent to reachy)
from reachy_start import *


# check whether the location of the object is within the reachable range
# x needs to be smaller or equal to 0.5m, but larger than 0.15m (edge of the table)
# y needs to be smaller than 0m but larger than -0.4m
# If the coordinates are outside of this range, it will return False
def check_execution_range(coord):
    x, y, z = coord
    valid = False

    if 0.5 >= x >= 0.15 and 0 >= y >= -0.4:
        valid = True

    return valid


# -------------------------- reachy's execution --------------------------------- #
def execute_reachy(reachy, intent, pickup_coord, place_coord):
    PICKUP = 'PickupObject'
    PUT = 'PutObject'
    GOTO = 'GotoLocation'
    POS1 = None
    POS2 = None
    POS1_ANGLE = 0
    POS2_ANGLE = 0

    print("Target position:     ", pickup_coord)
    print("Receptacle position: ", place_coord)

    if pickup_coord:
        if check_execution_range(pickup_coord):
            POS1 = pickup_coord

    if place_coord:
        if check_execution_range(place_coord):
            POS2 = place_coord

    # pick up and put down an object
    if PICKUP in intent and PUT in intent and pickup_coord:
        print("Perform pick up and place actions")

        # if the target object is reachable, reachy will pick up the object
        if POS1:
            # if the receptacle is reachable, reachy will place the target on top of the receptacle
            if POS2:
                print("picking up at {}, placing at {}".format(POS1, POS2))
                go_pick_go_place(reachy, (POS1, POS1_ANGLE), (POS2, POS2_ANGLE))
            # if the receptacle is not reachable, reachy will put the target back to its original place
            else:
                print("picking up at {}, placing at {}".format(POS1, POS1))
                go_pick_go_place(reachy, (POS1, POS1_ANGLE), (POS1, POS1_ANGLE))
        else:
            print("Target object is not reachable")
            start_and_rest(reachy)

    # pick up and hold for 3 seconds
    elif PICKUP in intent and not PUT in intent and POS1:
        go_to(reachy, (POS1, POS1_ANGLE), pick_action=True)

    # go to and do nothing
    elif intent == GOTO:
        go_to(reachy, (POS1, POS1_ANGLE), pick_action=False)

    else:
        print("Cannot find objects")
