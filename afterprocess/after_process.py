def remove_not_cloud_in_train_data(train_word_feq_set, err_index_end_list):
    ret_list = []
    for token_tag in err_index_end_list:
        if token_tag['token'] in train_word_feq_set:
            ret_list.append(token_tag)

    return ret_list
