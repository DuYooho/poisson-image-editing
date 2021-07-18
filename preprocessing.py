import numpy as np
import cv2 as cv

# 预处理类, 用于选择蒙版和复制点
class PreProcessing:
    def __init__(self):
        # 调用seamlessClone所需的两个成员
        self.selectedMask = None    # 蒙版灰度矩阵
                                    # 其中选中区为255, 非选中区值为0, 尺寸与src相同, 不包含边界
        self.selectedPoint = None   # 选择的复制位置
                                    # (注: 仅适用于自定义的seamlessClone, 若调用opencv的seamlessClone, x, y需要反转)
        # 中间变量, 私有
        self.__edgeList = []        # 选区边界点列表
        self.__preview = None       # 画线预览
        self.__edgeMat = None       # 用户所画线条灰度图
        self.__reMask = None        # 将蒙版扩张为目标图片大小
        self.__reImg = None         # 将选择图像扩张为目标图片大小
        self.__prePoint = None      # 上一次选点
        self.__minPoint = None      # 选点最小有效位置, 用于边界控制
        self.__maxPoint = None      # 选点最大有效位置, 用于边界控制

    # 左键拖动画线, 右键指定内部区域, 中键清除选择
    def __onMouseAction1(self, event, x, y, flags, param):
        # 选择起始点
        if event == cv.EVENT_LBUTTONDOWN:
            self.__edgeList = [(x, y)]
            self.__edgeMat = np.zeros(
                (param.shape[0], param.shape[1]), np.uint8)
            self.__preview = np.copy(param)
        # 画线
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            self.__edgeMat = cv.line(
                self.__edgeMat, self.__edgeList[-1], (x, y), 255)
            self.__preview = cv.line(
                self.__preview, self.__edgeList[-1], (x, y), (0, 0, 255))
            self.__edgeList.append((x, y))
            cv.imshow('select_mask', self.__preview)
        # 确定选区
        elif event == cv.EVENT_RBUTTONDOWN:
            retval, image, mask, rect = cv.floodFill(
                np.copy(self.__edgeMat), None, (x, y), 255)
            self.selectedMask = image - self.__edgeMat
            selectedImg = cv.copyTo(param, self.selectedMask)
            cv.imshow('select_mask', selectedImg)
        # 清空选区
        elif event == cv.EVENT_MBUTTONDOWN:
            self.__edgeList = []
            self.__edgeMat = np.zeros(
                (param.shape[0], param.shape[1]), np.uint8)
            self.__preview = np.copy(param)
            cv.imshow('select_mask', self.__preview)

    # 左键拖动调整目标位置
    def __onMouseAction2(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.__prePoint = (x, y)
        # 拖拽选区
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            # 鼠标点x表示列, y表示行, 矩阵处理时转换
            dx, dy = x-self.__prePoint[0], y-self.__prePoint[1]
            # 越界处理
            if self.selectedPoint[0] + dy < self.__minPoint[0] or self.selectedPoint[0] + dy > self.__maxPoint[0]:
                dy = 0
            if self.selectedPoint[1] + dx < self.__minPoint[1] or self.selectedPoint[1] + dx > self.__maxPoint[1]:
                dx = 0
            self.__prePoint = (self.__prePoint[0]+dx, self.__prePoint[1]+dy)
            self.selectedPoint = (
                self.selectedPoint[0]+dy, self.selectedPoint[1]+dx)
            # 移动
            self.__reMask = cv.warpAffine(self.__reMask, np.array(
                [[1, 0, dx], [0, 1, dy]], dtype=np.float64), (self.__reMask.shape[1], self.__reMask.shape[0]))
            self.__reImg = cv.warpAffine(self.__reImg, np.array(
                [[1, 0, dx], [0, 1, dy]], dtype=np.float64), (self.__reImg.shape[1], self.__reImg.shape[0]))
            _img = np.copy(param)
            _img[self.__reMask != 0] = 0
            _img = _img + self.__reImg
            cv.imshow('select_point', _img)

    # 选择蒙版和复制位置
    # 选择蒙版时, 左键拖动画线, 右键指定内部区域, 中键清除选择
    # 选择复制位置时, 左键拖动调整目标位置
    # 第一次按任意键确定蒙版选择, 第二次按任意键确定复制位置并退出
    def select(self, src, dst):
        # 选择蒙版
        cv.namedWindow('select_mask')
        cv.setMouseCallback('select_mask', lambda event, x, y, flags,
                            param: self.__onMouseAction1(event, x, y, flags, param), src)
        cv.imshow('select_mask', src)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # 结果处理
        pl = np.nonzero(self.selectedMask)
        selectedSrc = cv.copyTo(src, self.selectedMask)
        cutMask = self.selectedMask[np.min(
            pl[0])-1:np.max(pl[0])+2, np.min(pl[1])-1:np.max(pl[1])+2]
        cutSrc = selectedSrc[np.min(
            pl[0])-1:np.max(pl[0])+2, np.min(pl[1])-1:np.max(pl[1])+2]
        self.__minPoint = (
            (np.max(pl[0])-np.min(pl[0]))//2+1, (np.max(pl[1])-np.min(pl[1]))//2+1)
        # 选区过大
        if dst.shape[0] < cutMask.shape[0] or dst.shape[1] < cutMask.shape[1]:
            raise UserWarning
        # 选择复制位置
        cv.namedWindow('select_point')
        cv.setMouseCallback('select_point', lambda event, x, y, flags,
                            param: self.__onMouseAction2(event, x, y, flags, param), dst)
        # 初始化第一轮显示
        self.__reMask = np.zeros((dst.shape[0], dst.shape[1]))
        self.__reImg = np.zeros_like(dst)
        self.__reMask[:cutMask.shape[0],
                      :cutMask.shape[1]] = cutMask
        self.__reImg[:cutSrc.shape[0],
                     :cutSrc.shape[1]] = cutSrc
        self.selectedPoint = self.__minPoint[:]
        self.__maxPoint = (dst.shape[0]-cutMask.shape[0]+self.selectedPoint[0],
                           dst.shape[1]-cutMask.shape[1]+self.selectedPoint[1])
        _dst = np.copy(dst)
        _dst[self.__reMask != 0] = 0
        _dst = _dst + self.__reImg
        cv.imshow('select_point', _dst)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self.selectedMask, self.selectedPoint

    # 仅选择源图像区域
    def selectSrc(self, src):
        # 选择蒙版
        cv.namedWindow('select_mask')
        cv.setMouseCallback('select_mask', lambda event, x, y, flags,
                            param: self.__onMouseAction1(event, x, y, flags, param), src)
        cv.imshow('select_mask', src)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self.selectedMask
