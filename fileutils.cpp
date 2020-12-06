//
// Created by Sveta Morkva on 11/28/20.
//

#include "fileutils.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::vector<std::string> fileutils::get_file_list(const std::string &path) {
    std::vector<std::string> m_file_list;
    if (fs::is_directory(path)) {
        fs::path apk_path(path);
        fs::directory_iterator end;

        for (fs::directory_iterator i(apk_path); i != end; ++i) {
            const fs::path cp = (*i);
            if (cp.extension() == ".jpg" || cp.extension() == ".jpeg") {
                m_file_list.push_back(cp.string());
            }
        }
    }
    return m_file_list;
}

void fileutils::preprocessData() {
    const auto &trainFolder = "../images/train/";
    if (fs::exists(trainFolder)) {
        fs::path path_to_remove(trainFolder);
        for (fs::directory_iterator end_dir_it, it(path_to_remove); it!=end_dir_it; ++it) {
            fs::remove_all(it->path());
        }
    } else {
        fs::create_directory(trainFolder);
    }

    std::ofstream fileLabels;
    fileLabels.open("../metrics/train_data.csv", std::ofstream::out | std::ofstream::trunc);
    fileLabels << "Image_name, Correct label\n";
    const auto &book1Data = get_file_list("../images/Book1");
    const auto &book2Data = get_file_list("../images/Book2");
    const auto &noneData = get_file_list("../images/None");
    int num = 0;
    for (size_t i = 0; i < book1Data.size(); i++) {
        auto newfilename = trainFolder + std::to_string(num++) + ".jpg";
        fs::copy_file(book1Data[i], newfilename);
        fileLabels << newfilename << "," << 1 << "\n";
    }
    for (size_t i = 0; i < book2Data.size(); i++) {
        auto newfilename = trainFolder + std::to_string(num++) + ".jpg";
        fs::copy_file(book2Data[i], newfilename);
        fileLabels << newfilename << "," << 2 << "\n";
    }
    for (size_t i = 0; i < noneData.size(); i++) {
        auto newfilename = trainFolder + std::to_string(num++) + ".jpg";
        fs::copy_file(noneData[i], newfilename);
        fileLabels << newfilename << "," << 0 << "\n";
    }
    fileLabels.close();
}

std::string fileutils::getFilenameFromPath(const std::string &path) {
    auto it = path.rfind('/');
    return path.substr(it + 1);
}
