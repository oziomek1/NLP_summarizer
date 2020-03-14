# -*- coding: utf-8 -*-
import scrapy


class WnpSpider(scrapy.Spider):
    name = 'wnp'
    allowed_domains = ['wnp.pl']
    start_urls = ['http://wnp.pl/']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//ul[contains(@class, 'list-7')]//li//div[contains(@class, 'box-9')]//div//h3//a/@href"
    SUB_PAGE_URLS = "//div[contains(@class, 'help-nav')]//div[contains(@class, 'pageWidth')]//ul//li[contains(@class, 'has-sub')]//li//a[not(contains(@href, 'kto-jest-kim')) and not(contains(@href, 'katalog'))]/@href"
    ARTICLES_IN_BOXES_URLS = "//ul[contains(@class, 'list-7')]//li//ul[contains(@class, 'list-8')]//li//h3//a/@href"
    
    CATEGORY_MAIN_ARTICLE_URL = "//div[contains(@class, 'box-7')]//a/@href"
    CATEGORY_SIDE_ARTICLES_URLS = "//ul[contains(@class, 'list-17')]//li//a/@href"
    CATEGORY_RELATED_ARTICLES_URLS = "//ul[contains(@class, 'list-5')]//li//a//@href"

    TITLE_NODE = "//div[contains(@class, 'pre-article')]/h1/text()"
    LEAD_NODE = "//article/p/text()"
    TEXT_NODE = "//article/text()"
    TEXT_MAIN_POINTS = "//article/ul/li/strong/text()"

    def parse(self, response):
        # iterate through links to articles
        for quote in [
        	*response.xpath(self.ARTICLES_URLS),
        	*response.xpath(self.ARTICLES_IN_BOXES_URLS),
        	*response.xpath(self.CATEGORY_MAIN_ARTICLE_URL),
        	*response.xpath(self.CATEGORY_SIDE_ARTICLES_URLS),
        	*response.xpath(self.CATEGORY_RELATED_ARTICLES_URLS),
        ]:
            yield response.follow(quote, callback=self.parse_article)

        #iterate through next page links
        for next_page in response.xpath(self.SUB_PAGE_URLS):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
            'text_main_points': response.xpath(self.TEXT_MAIN_POINTS).extract(),
        }