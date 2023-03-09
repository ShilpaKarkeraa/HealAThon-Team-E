%Read an Image and show the the result of the detection
I = imread('355.jpg');
[bboxes,scores] = detect(detector,I);
if(~isempty(bboxes))
    percentage = scores * 100;
    label_str = sprintf("%s %4.2f %%", detector.ModelName, percentage);
    I = insertObjectAnnotation(I,'rectangle',bboxes,label_str);
    figure
    imshow(I)
end
%Evaluate the precision and the recall of the the detetcor using the testing set
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(detector, preprocessedTestData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

%Helper functions
function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
sz = size(data{1},[1 2]);
scale = targetSize(1:2)./sz;
data{1} = imresize(data{1},targetSize(1:2));

% Sanitize box data, if needed.
data{2} = helperSanitizeBoxes(data{2},sz);

% Resize boxes to new image size.
data{2} = bboxresize(data{2},scale);
end
%helperSanitizeBoxes Sanitize box data.
% This example helper is used to clean up invalid bounding box data. Boxes
% with values <= 0 are removed and fractional values are rounded to
% integers.
%
% If none of the boxes are valid, this function passes the data through to
% enable downstream processing to issue proper errors.

% Copyright 2020 The Mathworks, Inc.

function boxes = helperSanitizeBoxes(boxes, imageSize)
persistent hasInvalidBoxes
valid = all(boxes > 0, 2);
if any(valid)
    if ~all(valid) && isempty(hasInvalidBoxes)
        % Issue one-time warning about removing invalid boxes.
        hasInvalidBoxes = true;
        warning('Removing ground truth bouding box data with values <= 0.')
    end
    boxes = boxes(valid,:);
    boxes = roundFractionalBoxes(boxes, imageSize);
end

end

function boxes = roundFractionalBoxes(boxes, imageSize)
% If fractional data is present, issue one-time warning and round data and
% clip to image size.
persistent hasIssuedWarning

allPixelCoordinates = isequal(floor(boxes), boxes);
if ~allPixelCoordinates
    
    if isempty(hasIssuedWarning)
        hasIssuedWarning = true;
        warning('Rounding ground truth bounding box data to integer values.')
    end
    
    boxes = round(boxes);
    boxes(:,1:2) = max(boxes(:,1:2), 1); 
    boxes(:,3:4) = min(boxes(:,3:4), imageSize([2 1]));
end
end