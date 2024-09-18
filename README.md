# kp_matching_stuff_testing
testing on random image datasets


Im poor and i dont have gpu. 
```
docker build --platform linux/amd64 -t  keypoint_matching:latest .

docker run --platform linux/amd64 --rm -it \
    -v /Users/aryansingh/Downloads/tiny-imagenet-200:/data/tiny-imagenet-200 \
    -e DATA_PATH=/data/tiny-imagenet-200 \
    keypoint_matching:latest

```
