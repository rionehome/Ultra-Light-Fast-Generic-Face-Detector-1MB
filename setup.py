import os

from setuptools import setup

package_name = 'face_predictor'


def create_install_files(paths: list):
    target_data_files = [('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                         ('share/' + package_name, ['package.xml'])]

    for path in paths:
        for root, dirs, files in os.walk(path):
            print(root)
            target_data_files.append(('lib/{}/{}'.format(package_name, root), []))
            for file in files:
                print('{}/{}'.format(root, file))
                target_data_files[-1][1].append('{}/{}'.format(root, file))
    return target_data_files


# print(create_install_files(['vision', 'models']))

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'face_predictor.script',
    ],
    data_files=[('share/ament_index/resource_index/packages', ['resource/face_predictor']),
                ('share/face_predictor', ['package.xml']), ('lib/face_predictor/vision', ['vision/__init__.py']),
                ('lib/face_predictor/vision/utils', ['vision/utils/__init__.py',
                                                     'vision/utils/misc.py',
                                                     'vision/utils/box_utils.py',
                                                     'vision/utils/box_utils_numpy.py']),
                ('lib/face_predictor/vision/utils/__pycache__', []),
                ('lib/face_predictor/vision/transforms', ['vision/transforms/__init__.py',
                                                          'vision/transforms/transforms.py']),
                ('lib/face_predictor/vision/transforms/__pycache__', []),
                ('lib/face_predictor/vision/ssd', ['vision/ssd/data_preprocessing.py',
                                                   'vision/ssd/__init__.py',
                                                   'vision/ssd/mb_tiny_fd.py',
                                                   'vision/ssd/ssd.py',
                                                   'vision/ssd/predictor.py',
                                                   'vision/ssd/mb_tiny_RFB_fd.py']),
                ('lib/face_predictor/vision/ssd/__pycache__', []),
                ('lib/face_predictor/vision/ssd/config', ['vision/ssd/config/__init__.py',
                                                          'vision/ssd/config/fd_config.py']),
                ('lib/face_predictor/vision/ssd/config/__pycache__', []),
                ('lib/face_predictor/vision/nn', ['vision/nn/__init__.py',
                                                  'vision/nn/multibox_loss.py',
                                                  'vision/nn/mb_tiny_RFB.py',
                                                  'vision/nn/mb_tiny.py']),
                ('lib/face_predictor/vision/nn/__pycache__', []),
                ('lib/face_predictor/vision/__pycache__', []),
                ('lib/face_predictor/vision/datasets', ['vision/datasets/voc_dataset.py',
                                                        'vision/datasets/__init__.py']),
                ('lib/face_predictor/models', ['models/voc-model-labels.txt', 'models/readme']),
                ('lib/face_predictor/models/pretrained', ['models/pretrained/version-RFB-320.pth',
                                                          'models/pretrained/version-slim-320.pth',
                                                          'models/pretrained/version-RFB-640.pth',
                                                          'models/pretrained/version-slim-640.pth']),
                ('lib/face_predictor/models/onnx', ['models/onnx/version-slim-320.onnx',
                                                    'models/onnx/version-RFB-320.onnx',
                                                    'models/onnx/version-RFB-320_ncnn_slim.onnx',
                                                    'models/onnx/version-RFB-320_without_postprocessing.onnx',
                                                    'models/onnx/version-slim-320_without_postprocessing.onnx',
                                                    'models/onnx/version-slim-320_ncnn_slim.onnx']
                 )],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hirorittsu',
    maintainer_email='zq6295migly@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_predictor = face_predictor.script:main'
        ],
    },
)
