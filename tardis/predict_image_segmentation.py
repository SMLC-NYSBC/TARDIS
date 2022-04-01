def main():
    """
    Detect files in working dir
    build nn and load model
    build post-processing class

    For each file:
        - build temp dir
        - voxalize image to size
        - predict images patches
            - predict
            - threshold
            - save
        - stitch image
        - post-process / generate point cloud
        - save data
        - remove temp dir
"""
    pass


if __name__ == '__main__':
    main()
