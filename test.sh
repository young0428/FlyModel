# 하위 폴더들에서 _16frames 제거
cd ./model/wba_value_whole_features_16frames
for dir in piece_size_*_16frames; do
    newname=$(echo $dir | sed 's/_16frames//')
    mv "$dir" "$newname"
done