"""This script contains utility functions for 
generating tile plots of image/matrices.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def custom_tile_image_plot(
    layout,
    images,
    labels=None,
    filename="",
    cmap="gray",
    label_size=16,
    label_color=None,
    figure_size=(8., 8.),
):
    """
    Plots multiple images as subplots.
    
    Args:
        layout (tuple): Tuple of integers (m,n).
        images (np.array): NumPy array containing the images.
        labels (np.array): A list or NumPy array of labels (optional)
        filename (str): Filename to save the plot to as png (optional).
        cmap (str): Color map name (default is "gray"). 
        label_size (int): Font size of the labels.
        label_color (str or list): Label color(s). 
        figure_size (tuple): Figure size
    
    Returns:
        None
    """
    ## Options
    bottom_margin = 0.0
    left_margine  = 0.0
    vertical_spacing_fraction   = 0.05
    horizontal_spacing_fraction = 0.05
    #
    width  = (1.-left_margine)/(horizontal_spacing_fraction*(layout[0]-1)+layout[0])
    height = (1.-bottom_margin)/(vertical_spacing_fraction*(layout[1]-1)+layout[1])

    # Set figure size properly:
    fig = plt.figure(figsize=figure_size)

    axes_dict = {}
    for r in range(0,layout[1]):
        for c in range(0,layout[0]):
            index = (r*layout[1]+c)
            #
            axis_coord = [ left_margine  + c*(horizontal_spacing_fraction + 1.)*width,
                           bottom_margin + r*(vertical_spacing_fraction   + 1.)*height,  
                           width, 
                           height ]
            axes_dict[(r,c)] = plt.axes(axis_coord)
            axes_dict[(r,c)].set_aspect("equal")
            # axes_dict[(r,c)].get_xaxis().set_ticks([])  ## No x ticks
            # axes_dict[(r,c)].get_yaxis().set_ticks([])  ## No y ticks
            axes_dict[(r,c)].get_xaxis().set_visible(False)  ## Hide x axis
            axes_dict[(r,c)].get_yaxis().set_visible(False)  ## Hide y axis
            #
            if( labels is not None ):
                if(isinstance(label_color, str)):
                    plt.text(0.1, 0.1, str(labels[index]), 
                        horizontalalignment="center",
                        verticalalignment="center", 
                        transform=axes_dict[(r,c)].transAxes,
                        rotation=0.,
                        rotation_mode="anchor",
                        color=label_color,
                        alpha=.85,
                        fontsize=label_size)
                elif(isinstance(label_color, list)):
                    plt.text(0.1, 0.1, str(labels[index]), 
                        horizontalalignment="center",
                        verticalalignment="center", 
                        transform=axes_dict[(r,c)].transAxes,
                        rotation=0.,
                        rotation_mode="anchor",
                        color=label_color[(r*layout[1]+c)],
                        alpha=.75,
                        fontsize=label_size)
                else:
                    print(f"WARNING: \"label_color\" is neither a str nor a list...")
                    plt.text(0.1, 0.1, str(labels[index]), 
                        horizontalalignment="center",
                        verticalalignment="center", 
                        transform=axes_dict[(r,c)].transAxes,
                        rotation=0.,
                        rotation_mode="anchor",
                        color="green",
                        alpha=.85,
                        fontsize=label_size)
            #
            image = images[(r*layout[1]+c),:]
            if(len(image.shape) == 2):
                axes_dict[(r,c)].imshow(image, origin="upper", cmap=cmap)
            elif(image.shape[-1] == 3):
                axes_dict[(r,c)].imshow(image, origin="upper")
            else:
                raise ValueError(
                    f"Expected either a grayscale (2D) or RGB image!\n{image.shape}")

    #
    if( filename!="" ):
        plt.savefig(filename, dpi=100, bbox_inches="tight")
    else:
        plt.show()
    
    
def custom_tile_plot_with_inference_hists(
    layout,
    images,
    labels,
    predictions,
    classes=np.linspace(start=0,stop=10,num=10,endpoint=False,dtype=np.uint8),
    only_mispredicted=False,
    filename="", 
    cmap="gray", 
    label_size=32
):
    """
    Show multiple images AND their class probabilities as subplots.
    
    Args:
        layout (tuple): Tuple of integers (m,n).
        data (np.array): NumPy array containing the images.
        labels (np.array): A list or NumPy array of labels.
        predictions (np.array): Contains the inference class probabilities.
        classes (np.array): NumPy array of classes (optional -- default is [0..10])
        only_mispredicted (bool): If True, will skip correctly predicted ones.
        filename (str): Filename to save the plot to as PNG (optional).
        cmap (str): Color map name (default is "gray"). 
        label_size (int): Font size of the labels.
    
    Returns:
        None
    """
    ## Options
    bottom_margin = 0.0
    left_margine  = 0.0
    vertical_spacing_fraction    = 0.05
    horizontal_spacing_fraction  = 0.05
    realtive_image_bar_axes_size = 0.32
    #
    ## Identify misclassified ones if specified:
    if( only_mispredicted ):
        images_      = np.copy(images)
        predictions_ = np.copy(predictions)
        labels_      = np.copy(labels)
        mispred_indices = np.where( np.argmax(predictions_,axis=1)==labels_, False, True)
        images_      = images_[mispred_indices]
        labels_      = labels_[mispred_indices]
        predictions_ = predictions_[mispred_indices]
    else:
        images_      = images
        labels_      = labels
        predictions_ = predictions
    #
    ## Cell/Tile width and height
    width  = (1.-left_margine)/(horizontal_spacing_fraction*(layout[0]-1)+layout[0])  ## Width of a cell containing two plots
    height = (1.-bottom_margin)/(vertical_spacing_fraction*(layout[1]-1)+layout[1])   ## Height of a cell containing two plots
    #
    ## Width and height for `imshow` and bar(h) plots:
    if( width>=height ):
        horizontal_layout = True
        if( width-height>=realtive_image_bar_axes_size*width ):
            image_width  = height
            image_height = height
            image_bottom_delta = 0.
            image_left_delta   = (width-height)
            #
            bar_width  = (width-height)
            bar_height = height
            bar_bottom_delta = 0.
            bar_left_delta   = 0.
        else:
            delta_ = realtive_image_bar_axes_size*width - (width-height) 
            image_width  = height-delta_
            image_height = height-delta_
            image_bottom_delta = .5*(height-delta_)
            image_left_delta   = realtive_image_bar_axes_size*width
            #
            bar_width  = realtive_image_bar_axes_size*width
            bar_height = height-delta_
            bar_bottom_delta = .5*(height-delta_)
            bar_left_delta   = 0.
    elif( width<height ):
        horizontal_layout = False
        if( height-width>=realtive_image_bar_axes_size*height ):
            image_width  = width
            image_height = width
            image_bottom_delta = (height-width)
            image_left_delta   = 0.
            #
            bar_width  = width
            bar_height = (height-width)
            bar_bottom_delta = 0.
            bar_left_delta   = 0.
        else:
            delta_ = realtive_image_bar_axes_size*height - (height-width)
            image_width  = width-delta_
            image_height = width-delta_
            image_bottom_delta = realtive_image_bar_axes_size*height
            image_left_delta   = .5*(width-delta_)
            #
            bar_width  = width-delta_
            bar_height = realtive_image_bar_axes_size*height
            bar_bottom_delta = 0.
            bar_left_delta   = .5*(width-delta_) 
    #
    ## Set figure size properly:
    fig_width  = 10. ## inches
    fig_height = 10. ## inches
    plt.figure( figsize=(fig_width,fig_height) )
    #
    image_axes_dict = {}
    bar_axes_dict   = {}
    for r in range(0,layout[1]):
        for c in range(0,layout[0]):
            index = (r*layout[1]+c)  
            highest_prob_label = np.where( predictions_[index,:]==np.amax(predictions_[index,:]) )                             
            #
            ## Image
            cell_corner = ( left_margine  + c*(horizontal_spacing_fraction + 1.)*width,
                            bottom_margin + r*(vertical_spacing_fraction   + 1.)*height )
            image_axis_coord = [ image_left_delta   + cell_corner[0], 
                                 image_bottom_delta + cell_corner[1],  
                                 image_width, 
                                 image_height ]
            image_axes_dict[(r,c)] = plt.axes(image_axis_coord)
            image_axes_dict[(r,c)].set_aspect("equal")
            image_axes_dict[(r,c)].get_xaxis().set_visible(False)  ## Hide x axis
            image_axes_dict[(r,c)].get_yaxis().set_visible(False)  ## Hide y axis
            image_axes_dict[(r,c)].imshow( images_[index,:], origin="upper", cmap=cmap)
            #
            ## Prediction
            bar_axis_coord = [ bar_left_delta   + cell_corner[0], 
                               bar_bottom_delta + cell_corner[1],  
                               bar_width, 
                               bar_height ] 
            bar_axes_dict[(r,c)] = plt.axes(bar_axis_coord)
            bar_axes_dict[(r,c)].set_aspect("auto") 
            if( horizontal_layout ):
                bar_axes_dict[(r,c)].set_xticks(np.arange(0,1.1,step=.25))
                bar_axes_dict[(r,c)].tick_params(axis="x", labelsize=6., labelrotation=90)
                bar_axes_dict[(r,c)].set_yticks(classes)
                bar_axes_dict[(r,c)].tick_params(axis="y", labelsize=6., labelrotation=0)
                if( highest_prob_label == labels_[index] ):
                    bar_axes_dict[(r,c)].barh( classes, predictions_[index,:], color="b")
                    bar_axes_dict[(r,c)].set_facecolor((.9, .9, 1.))
                else:
                    bar_axes_dict[(r,c)].barh( classes, predictions_[index,:], color="r")
                    bar_axes_dict[(r,c)].set_facecolor((1., .9, .9))
            else:
                bar_axes_dict[(r,c)].set_yticks(np.arange(0,1.1,step=.25))
                bar_axes_dict[(r,c)].tick_params(axis="y", labelsize=6., labelrotation=0)
                bar_axes_dict[(r,c)].set_xticks(classes)
                bar_axes_dict[(r,c)].tick_params(axis="x", labelsize=6., labelrotation=90)
                if( highest_prob_label == labels_[index] ):
                    bar_axes_dict[(r,c)].bar( classes, predictions_[index,:], color="b")
                    bar_axes_dict[(r,c)].set_facecolor((.9, .9, 1.))
                else:
                    bar_axes_dict[(r,c)].bar( classes, predictions_[index,:], color="r")
                    bar_axes_dict[(r,c)].set_facecolor((1., .9, .9))
            #
            if( c!=0 ):
                bar_axes_dict[(r,c)].get_yaxis().set_visible(False)
            if( r!=0 ):
                bar_axes_dict[(r,c)].get_xaxis().set_visible(False)
            #
            bar_axes_dict[(r,c)].text(.5, .5, str(labels_[index]),
                transform=bar_axes_dict[(r,c)].transAxes,
                verticalalignment="center", 
                horizontalalignment="center",
                color="black",
                alpha=.5,
                fontsize=label_size)
    #
    if( filename!="" ):
        plt.savefig(filename, dpi=100, bbox_inches="tight")
    else:
        plt.show()