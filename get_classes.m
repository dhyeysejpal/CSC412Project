dict = struct();

dict.('angry') = {};
dict.('sad') = {};
dict.('disgust') = {};
dict.('fear') = {};
dict.('happy') = {};
dict.('surprise') = {};
dict.('neutral') = {};


data = load('labeled_images.mat');

fnames = fieldnames(dict);

for i = 1:2925
    l = data.tr_labels(i);
    label = char(fnames(l));
    dict.(label) = [dict.(label), data.tr_images(:,:,i)];
end

angry = dict.angry;
save('angry.mat', 'angry', '-mat');
