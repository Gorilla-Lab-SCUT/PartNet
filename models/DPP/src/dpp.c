#include <TH/TH.h>
#include <math.h>
#include <stdio.h>

int dpp_forward(int square_size, int proposals_per_square, int proposals_per_image, int spatital_scale, THFloatTensor * box_plan,
                    THFloatTensor * histogram, THFloatTensor * score_sum, THFloatTensor * output, THFloatTensor * features)
{
    // Grab the input tensor
    float * box_plan_flat = THFloatTensor_data(box_plan);
    float * histogram_flat = THFloatTensor_data(histogram);
    float * score_sum_flat = THFloatTensor_data(score_sum);
    float * output_flat = THFloatTensor_data(output);
    float * features_flat = THFloatTensor_data(features);

    int batch_size = THFloatTensor_size(features, 0);

    int data_height = THFloatTensor_size(features, 2);
    // data width
    int data_width = THFloatTensor_size(features, 3);
    // Number of channels
    int num_channels = THFloatTensor_size(features, 1);

    int b, c, h, w, max_h, max_w, max_value, p;
    int index_proposals = 0;

    for (b = 0; b < batch_size; b++)
    {
        // printf("here is the batch of  %d", b);
        THFloatStorage_fill(THFloatTensor_storage(histogram), 0);
        // printf("here is the line after fill    lalalla  ");
        THFloatStorage_fill(THFloatTensor_storage(score_sum), 0);

        for (c =0; c < num_channels; c++)
        {
            // printf("histogram of  %d", c);
            //find the max position for each channel, and add 1 to the histogram
            const int index_features = b * data_height * data_width * num_channels + c * data_height * data_width;
            max_w = 0;
            max_h = 0;
            max_value = 0;

            for (h=0; h < data_height; h++)
            {
                for (w=0; w< data_width; w++)
                {
                    const int index_histogram = h * data_width + w;
                    if(features_flat[index_features + index_histogram] > max_value)
                    {
                        max_value = features_flat[index_features + index_histogram];
                        max_w = w;
                        max_h = h;
                    }
                }
            }
            histogram_flat[max_h * data_width + max_w] += 1;
        }

        /// calculate the score sum
        for (c =0; c < num_channels; c++)
        {
            // printf("score sum of  %d", c);
            //add values at each position of the feature to the score sum
            const int index_features = b * data_height * data_width * num_channels + c * data_height * data_width;

            for (h=0; h < data_height; h++)
            {
                for (w=0; w< data_width; w++)
                {
                    const int index_score_sum = h *data_width + w;
                    score_sum_flat[index_score_sum] += features_flat[index_features + index_score_sum];
                }
            }
        }

        //
        int sub_h, sub_w;
        for (h=0; h<data_height; h=h+square_size)
        {
            // printf("here is the height of  %d \n", h);
            for(w=0; w<data_width; w=w+square_size)
            {
                max_w = 0;
                max_h = 0;
                max_value = 0;
                // printf("here is the width of  %d \n", w);
                for (sub_h=h; sub_h<h+square_size; sub_h++)
                {
                    //printf("here is the sub height of  %d \n", sub_h);
                    for (sub_w=w; sub_w<w+square_size; sub_w++)
                    {
                        // printf("test if it is the max value \n");
                        if (max_value < histogram_flat[sub_h * data_width + sub_w])
                        {
                            max_value = histogram_flat[sub_h * data_width + sub_w];
                            max_w = sub_w;
                            max_h = sub_h;
                        }
                    }
                }
                // printf("find the position of max value \n");
                if (max_value == 0)
                {
                    for (sub_h=h; sub_h<h+square_size; sub_h++)
                    {
                        for (sub_w=w; sub_w<w+square_size; sub_w++)
                        {
                            if (max_value < score_sum_flat[sub_h * data_width + sub_w])
                            {
                                max_value = score_sum_flat[sub_h * data_width + sub_w];
                                max_w = sub_w;
                                max_h = sub_h;
                            }
                        }
                    }
                }
                //printf("begin to generate proposals");
                int index_box = 0;
                for (p=0; p < proposals_per_square; p++)
                {
                    output_flat[index_proposals] = b;
                    output_flat[index_proposals + 1] = fmaxf((max_w + box_plan_flat[index_box]) * spatital_scale, 0);
                    output_flat[index_proposals + 2] = fmaxf((max_h + box_plan_flat[index_box + 1]) * spatital_scale, 0);
                    output_flat[index_proposals + 3] = fminf((max_w + box_plan_flat[index_box + 2]) * spatital_scale, 448);
                    output_flat[index_proposals + 4] = fminf((max_h + box_plan_flat[index_box + 3]) * spatital_scale, 448);
                    index_proposals += 5;
                    index_box += 4;
                }

            }
        }
    }
    return 1;
}
